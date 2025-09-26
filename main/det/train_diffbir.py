import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
import utils.filter_warning

import torch
from copy import deepcopy
from tqdm import tqdm
from model import SwinIR, ControlLDM, Diffusion
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.common import (
    instantiate_from_config, load_network,
    calculate_psnr_pt, wavelet_reconstruction
)
from utils.detection import (
    GroupedBatchSampler, CocoEvaluator, prepare_environment, prepare_batch,
    create_aspect_ratio_groups, get_coco_api_from_dataset, batch_to_list, 
    collate_fn, _get_iou_types, draw_box, suppress_stdout
)
from utils.sampler import SpacedSampler
from argparse import ArgumentParser
from omegaconf import OmegaConf
from accelerate import Accelerator, DataLoaderConfiguration
from torchvision.utils import make_grid, save_image


def main(args) -> None:
    # setup environment
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cfg = OmegaConf.load(args.config)
    accelerator = Accelerator(dataloader_config=DataLoaderConfiguration(split_batches=True),
                              mixed_precision=cfg.train.precision)
    device = accelerator.device
    dirs, Logging = prepare_environment(__name__, cfg, args, accelerator)
    exp_dir, ckpt_dir, img_dir = dirs["exp"], dirs["ckpt"], dirs["img"]
    is_coco = True if cfg.dataset.get('is_coco') else False
    
    # create and load models
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    swinir.load_state_dict(torch.load(cfg.train.resume_swinir, map_location="cpu"), strict=True)
    Logging(f"Load SwinIR weight from checkpoint: {cfg.train.resume_swinir}")
    
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    cldm.load_pretrained_sd(sd)
    if cfg.train.resume_cldm:
        cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume_cldm, map_location="cpu"))
        Logging(f"Load ControlNet weight from checkpoint: {cfg.train.resume_cldm}")
    else:
        cldm.load_controlnet_from_unet()
        Logging(f"Load ControlNet weight from pretrained SD weight")
    
    teacher_detnet = instantiate_from_config(cfg.model.teacher_detnet)  # detnet trained on HQ images   
    load_path = cfg.train.resume_teacher_detnet
    teacher_detnet = load_network(teacher_detnet, load_path, strict=cfg.train.strict_load)
    for p in teacher_detnet.parameters(): p.requires_grad = False  # will not be trained
    Logging(f"Load Teacher DetectionNetwork weight from checkpoint: {load_path}, strict: {cfg.train.strict_load}")
    
    detnet = instantiate_from_config(cfg.model.detnet)
    if cfg.train.resume_detnet:
        detnet = load_network(detnet, cfg.train.resume_detnet, strict=cfg.train.strict_load)
        Logging(f"Load DetectionNetwork weight from checkpoint: {cfg.train.resume_detnet}, strict: {cfg.train.strict_load}")
    else:
        Logging("Initialize DetectionNetwork from scratch")
        
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    sampler = SpacedSampler(diffusion.betas)
    
    # setup optimizer and scheduler
    diffbir_params = list(cldm.controlnet.parameters())
    opt_diffbir = torch.optim.AdamW(
        diffbir_params, lr=cfg.train.learning_rate_diffbir
    )
    opt_detnet = torch.optim.SGD(
        detnet.parameters(), lr=cfg.train.learning_rate_detnet,
        momentum=0.9, weight_decay=1e-4
    )
    sch_diffbir = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_diffbir, T_max=cfg.train.train_steps,
        eta_min=1e-7
    )
    sch_detnet = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_detnet, T_max=cfg.train.train_steps,
        eta_min=1e-7
    )
    
    # setup data
    with suppress_stdout():
        dataset = instantiate_from_config(cfg.dataset.train)
    train_sampler = torch.utils.data.RandomSampler(dataset)
    group_ids = create_aspect_ratio_groups(dataset, k=cfg.train.aspect_ratio_group_factor, Logging=Logging)
    batch_sampler = GroupedBatchSampler(train_sampler, group_ids, cfg.train.batch_size)
    loader = DataLoader(
        dataset=dataset, batch_sampler=batch_sampler,
        num_workers=cfg.train.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    batch_transform = None
    if cfg.dataset.get("batch_transform"):
        batch_transform = instantiate_from_config(cfg.dataset.batch_transform)
    Logging(f"Training dataset contains {len(dataset):,} images from {dataset.root}")
    
    if cfg.val.batch_size == -1:
        num_gpus = accelerator.state.num_processes
        cfg.val.batch_size = num_gpus
        Logging(f"Validation batch size is automatically set to the number of available GPUs: {num_gpus}")
    with suppress_stdout():
        val_dataset = instantiate_from_config(cfg.dataset.val)    
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=cfg.val.batch_size, shuffle=False,
        num_workers=cfg.val.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    Logging(f"Validation dataset contains {len(val_dataset):,} images from {val_dataset.root}")

    # prepare models, evaluator, training logs
    swinir.eval().to(device)
    cldm.train().to(device)
    teacher_detnet.eval().to(device)
    detnet.train().to(device)
    diffusion.to(device)
    swinir, cldm, teacher_detnet, detnet, opt_diffbir, opt_detnet, sch_diffbir, sch_detnet, loader, val_loader = \
        accelerator.prepare(swinir, cldm, teacher_detnet, detnet, opt_diffbir, opt_detnet, sch_diffbir, sch_detnet, loader, val_loader)
    pure_swinir = accelerator.unwrap_model(swinir)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    pure_detnet = accelerator.unwrap_model(detnet)
    frozen_parameters = []  # to control the only trainable detection network parameters
    for n, p in detnet.named_parameters():
        if not p.requires_grad:
            frozen_parameters += [n]
    Logging("Preparing COCO API for evaluation...")
    with suppress_stdout():
        coco = get_coco_api_from_dataset(val_dataset)
    iou_types = _get_iou_types(pure_detnet)
    global_step = 0
    max_steps = cfg.train.train_steps
    epoch = 0
    loss_records = dict(eps=[], det=[], fm=[])
    loss_avg_records = deepcopy(loss_records)
    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
    
    # Training:
    Logging(f"Training for {max_steps} steps...")
    while global_step < max_steps:
        pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch", total=len(loader))
        for batch in loader:
            gt_list, _, gt_batch, lq_batch, annot_list, _, bs = prepare_batch(batch, device, batch_transform)
            prompt = [cfg.train.default_prompt] * bs
            
            with torch.no_grad():
                pre_res_batch = swinir(lq_batch)
            
            # Train diffbir:
            with accelerator.autocast():
                cldm.train()
                
                with torch.no_grad():
                    z_0 = pure_cldm.vae_encode(gt_batch * 2 - 1)
                    cond = pure_cldm.prepare_condition(pre_res_batch, prompt)
                    t = torch.randint(0, diffusion.num_timesteps, (bs,), device=device)
                loss_eps = diffusion.p_losses(cldm, z_0, t, cond)
            
            opt_diffbir.zero_grad()
            accelerator.backward(loss_eps)
            opt_diffbir.step(), sch_diffbir.step()
            
            # Train detnet:
            with accelerator.autocast():
                cldm.eval()
                bs2 = bs//2  # For training detnet, use 1/2 batch from DiffBIR and 1/2 from HQ images
                # NOTE: This is the same setting as done in EDTR training
                
                # restoration
                with torch.no_grad():
                    cond2 = dict(c_txt=cond["c_txt"][:bs2], c_img=cond["c_img"][:bs2])
                    
                    z = sampler.sample(
                        model=cldm, device=device, steps=50, batch_size=bs2, x_size=z_0.shape[1:],
                        cond=cond2, uncond=None, cfg_scale=1.0, x_T=None,
                        progress=accelerator.is_local_main_process, progress_leave=False
                    )
                    res_batch = wavelet_reconstruction((pure_cldm.vae_decode(z) + 1) / 2, pre_res_batch[:bs2])
                    res_list = batch_to_list(res_batch, gt_list[:bs2])
                    
                # task loss
                _, loss_dict, feat_student = detnet(res_list + gt_list[bs2:], annot_list, return_feat=True)
                loss_det = sum(loss_value for loss_value in loss_dict.values()) * cfg.train.weight_det
                
                # feature-matching loss
                with torch.no_grad():
                    _, _, feat_teacher = teacher_detnet(gt_list, return_feat=True)
                loss_fm = (F.l1_loss(input=feat_student['features']['0'], target=feat_teacher['features']['0'], reduction="mean") * 0.5 + \
                    F.l1_loss(input=feat_student['features']['1'], target=feat_teacher['features']['1'], reduction="mean") * 0.5) * cfg.train.weight_fm
            
            opt_detnet.zero_grad()
            accelerator.backward(loss_det + loss_fm)
            opt_detnet.step(), sch_detnet.step()

            accelerator.wait_for_everyone()

            global_step += 1
            loss_records["eps"].append(loss_eps.item())
            loss_records["det"].append(loss_det.item())
            loss_records["fm"].append(loss_fm.item())
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch:04d}, Steps: {global_step:07d}, Eps: {loss_eps.item():.3f}, DET: {loss_det.item():.3f}, FM: {loss_fm.item():.3f}")

            # log training loss, learning rate
            if global_step % cfg.train.log_every == 0 or (args.debug):
                for key in loss_records.keys():
                    loss_avg_records[key] = accelerator.gather(torch.tensor(loss_records[key], device=device).unsqueeze(0)).mean().item()
                    loss_records[key].clear()
                
                if accelerator.is_local_main_process:
                    loss_summary = f"[{global_step:05d}/{max_steps:05d}] Training loss: ( " + \
                        ", ".join([f"{key}: {value:.4f}" for key, value in loss_avg_records.items()]) + " )"
                    Logging(loss_summary, print=False)
                    for key in loss_records.keys():
                        writer.add_scalar("loss/loss_{}".format(key), loss_avg_records[key], global_step)
                    
                    writer.add_scalar("train/learning_rate_diffbir", opt_diffbir.param_groups[0]['lr'], global_step)
                    writer.add_scalar("train/learning_rate_detnet", opt_detnet.param_groups[0]['lr'], global_step)

            # save checkpoint
            if global_step % cfg.train.ckpt_every == 0 or (args.debug):
                if accelerator.is_local_main_process:
                    torch.save(pure_cldm.controlnet.state_dict(), f"{ckpt_dir}/cldm_{global_step:07d}.pt")
                    torch.save(pure_detnet.state_dict(), f"{ckpt_dir}/detnet_{global_step:07d}.pt")

            # save images
            if global_step % cfg.train.image_every == 0 or global_step == 1 or (args.debug):
                with torch.no_grad():
                    N = 4
                    log_gt, log_lq, log_pre_res = gt_batch[:N], lq_batch[:N], pre_res_batch[:N]
                    log_cond = {k:v[:N] for k, v in cond.items()}
                    cldm.eval()
                    z = sampler.sample(
                        model=cldm, device=device, steps=50, batch_size=len(log_gt), x_size=z_0.shape[1:],
                        cond=log_cond, uncond=None, cfg_scale=1.0, x_T=None,
                        progress=accelerator.is_local_main_process, progress_leave=False
                    )
                    log_res = wavelet_reconstruction((pure_cldm.vae_decode(z) + 1) / 2, log_pre_res)
                    
                    if accelerator.is_local_main_process:
                        for tag, image in [
                            ("image/gt", log_gt), ("image/lq", log_lq),
                            ("image/pre_restored", log_pre_res), ("image/restored", log_res),
                        ]:
                            grid_image = make_grid(image, nrow=4)
                            writer.add_image(tag, grid_image, global_step)
                            img_name = '{}_{:06d}.png'.format(os.path.basename(tag), global_step)
                            save_image(grid_image, os.path.join(img_dir, img_name))
                        del grid_image
                    cldm.train()
            
            # evaluation
            if global_step % cfg.val.val_every == 0 or (args.debug):
                cldm.eval(), detnet.eval()
                coco_evaluator = CocoEvaluator(coco, iou_types)
                val_psnr = []
                val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch",
                                total=len(val_loader), leave=False, desc="Validation")
                for val_batch in val_loader:
                    val_gt_list, _, _, val_lq_batch, val_annot_list, _, val_bs = prepare_batch(val_batch, device)
                    val_prompt = [cfg.val.default_prompt] * val_bs
                    assert (val_bs == 1)
                    
                    with torch.no_grad():
                        # restoration
                        val_pre_res_batch = swinir(val_lq_batch)
                        val_cond = pure_cldm.prepare_condition(val_pre_res_batch, val_prompt)    
                        val_z = sampler.sample(
                            model=cldm, device=device, steps=50, batch_size=val_bs, x_size=z_0.shape[1:],
                            cond=val_cond, uncond=None, cfg_scale=1.0, x_T=None,
                            progress=accelerator.is_local_main_process, progress_leave=False
                        )
                        val_res_batch = wavelet_reconstruction((pure_cldm.vae_decode(val_z) + 1) / 2, val_pre_res_batch)
                        val_res_list = batch_to_list(val_res_batch, val_gt_list)
                        
                        # detection
                        val_pred_list, _ = detnet(val_res_list)
                        val_pred_list = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in val_pred_list]
                        
                        # calculate metrics
                        val_gt, val_res = val_gt_list[0].unsqueeze(0), val_res_list[0].unsqueeze(0)
                        val_psnr_batch = calculate_psnr_pt(val_res, val_gt, crop_border=0).unsqueeze(0)
                        val_psnr_batch = accelerator.gather_for_metrics(val_psnr_batch)
                        res = {annot["image_id"]: pred for annot, pred in zip(val_annot_list, val_pred_list)}
                        
                        if accelerator.is_local_main_process:
                            for v in val_psnr_batch: val_psnr += [v.item()]
                        coco_evaluator.update(res)
                        accelerator.wait_for_everyone()
                        
                    val_pbar.update(1)
                val_pbar.close()
                
                coco_evaluator.synchronize_between_processes()
                with suppress_stdout():
                    coco_evaluator.accumulate()
                
                if accelerator.is_local_main_process:
                    avg_val_psnr = torch.tensor(val_psnr).mean().item()
                    det_results = coco_evaluator.summarize(Logging=Logging)
                    for tag, val in [
                        ("val/psnr", avg_val_psnr),
                        ("val/mAP@[0.5:0.95]", det_results["mAP@[0.5:0.95]"]),
                        ("val/mAP@0.5", det_results["mAP@0.5"]),
                    ]:
                        Logging(f"{tag}: {val:.1f}")
                        writer.add_scalar(tag, val, global_step)
                cldm.train(), detnet.train()
            
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break
        
        pbar.close()
        epoch += 1

    # save the last model weight
    if accelerator.is_local_main_process:
        torch.save(pure_swinir.state_dict(), f"{ckpt_dir}/swinir_last.pt")
        torch.save(pure_cldm.controlnet.state_dict(), f"{ckpt_dir}/cldm_last.pt")
        torch.save(pure_detnet.state_dict(), f"{ckpt_dir}/detnet_last.pt")
        Logging("Done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(args)
