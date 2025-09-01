import os
import sys
import torch
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from copy import deepcopy
from tqdm import tqdm
from model import SwinIR
from einops import rearrange
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.common import instantiate_from_config, load_network, copy_opt_file, set_logger, print_attn_type, calculate_psnr_pt
from utils.detection import (
    GroupedBatchSampler, CocoEvaluator, draw_box, create_aspect_ratio_groups, collate_fn,
    get_coco_api_from_dataset, _get_iou_types, list_to_batch, batch_to_list
)
from argparse import ArgumentParser
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision.utils import make_grid, save_image


def main(args) -> None:
    # Setup accelerator
    cfg = OmegaConf.load(args.config)
    accelerator = Accelerator(split_batches=True, mixed_precision=cfg.train.precision)
    set_seed(231)
    device = accelerator.device
    
    def Logging(text, print=True):
        if accelerator.is_local_main_process:
            if print:
                logger.info(text)
            else:
                logger.debug(text)

    # Set up the experiment folder and logger
    if accelerator.is_local_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        img_dir = os.path.join(exp_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")
        logger = set_logger(__name__, exp_dir, logger_name="logger.log")
        copy_opt_file(args.config, exp_dir)
        print_attn_type(Logging=Logging)
        Logging(f"Random seed: {cfg.train.seed}")
        Logging("Using xformers attention as default") 

    # Create models
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    if cfg.train.resume_swinir:
        swinir.load_state_dict(torch.load(cfg.train.resume_swinir, map_location="cpu"), strict=True)
        Logging(f"Load SwinIR weight from checkpoint: {cfg.train.resume_swinir}")
    else:
        Logging("Initialize SwinIR from scratch")
    
    detnet = instantiate_from_config(cfg.model.detnet)
    if cfg.train.resume_detnet:
        detnet = load_network(detnet, cfg.train.resume_detnet, strict=cfg.train.strict_load)
        Logging(f"Load DetectionNetwork weight from checkpoint: {cfg.train.resume_detnet}, strict: {cfg.train.strict_load}")
    else:
        Logging("Initialize DetectionNetwork from scratch")
    
    # Setup optimizer:
    opt_swinir = torch.optim.AdamW(
        swinir.parameters(), lr=cfg.train.learning_rate_swinir,
        weight_decay=0
    )
    
    opt_detnet = torch.optim.SGD(
        detnet.parameters(), lr=cfg.train.learning_rate_detnet,
        momentum=0.9, weight_decay=1e-4
    )
    
    # Setup scheduler:
    sch_swinir = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_swinir, T_max=cfg.train.train_steps,
        eta_min=1e-7
    )
    
    sch_detnet = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_detnet, T_max=cfg.train.train_steps,
        eta_min=1e-7
    )
    
    # Setup data
    dataset = instantiate_from_config(cfg.dataset.train)
    train_sampler = torch.utils.data.RandomSampler(dataset)
    group_ids = create_aspect_ratio_groups(dataset, k=cfg.train.aspect_ratio_group_factor, Logging=Logging)
    batch_sampler = GroupedBatchSampler(train_sampler, group_ids, cfg.train.batch_size)
    loader = DataLoader(
        dataset=dataset, batch_sampler=batch_sampler,
        num_workers=cfg.train.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    Logging(f"Training dataset contains {len(dataset):,} images from {dataset.root}")
    
    if cfg.val.batch_size == -1:
        num_gpus = accelerator.state.num_processes
        cfg.val.batch_size = num_gpus
        Logging(f"Validation batch size is automatically set to the number of available GPUs: {num_gpus}")
    val_dataset = instantiate_from_config(cfg.dataset.val)    
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=cfg.val.batch_size, shuffle=False,
        num_workers=cfg.val.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    Logging(f"Validation dataset contains {len(val_dataset):,} images from {val_dataset.root}")

    # Prepare models for training
    swinir.train().to(device)
    detnet.train().to(device)
    swinir, detnet, opt_swinir, opt_detnet, sch_swinir, sch_detnet, loader, val_loader = \
        accelerator.prepare(swinir, detnet, opt_swinir, opt_detnet, sch_swinir, sch_detnet, loader, val_loader)
    pure_swinir = accelerator.unwrap_model(swinir)
    pure_detnet = accelerator.unwrap_model(detnet)
    frozen_parameters = []
    for n, p in detnet.named_parameters():
        if not p.requires_grad:
            frozen_parameters += [n]

    # Define variables related to training
    global_step = 0
    max_steps = cfg.train.train_steps
    epoch = 0
    loss_records = dict(swinir_pix=[], swinir_tdp=[], detnet=[])
    loss_avg_records = deepcopy(loss_records)
    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
    
    # Training
    Logging("Preparing COCO API for evaluation...")
    coco = get_coco_api_from_dataset(val_dataset)
    iou_types = _get_iou_types(pure_detnet)
    if accelerator.mixed_precision == 'fp16':
        Logging("Mixed precision is applied")
    Logging(f"Training for {max_steps} steps...")
    
    while global_step < max_steps:
        pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch", total=len(loader))
        for gt_list, lq_list, annot_list, path_list in loader:
            gt_list = list(rearrange(gt, 'h w c -> c h w').contiguous().float().to(device) for gt in gt_list)
            lq_list = list(rearrange(lq, 'h w c -> c h w').contiguous().float().to(device) for lq in lq_list)
            annot_list = [{k: v.long().to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in annot_list]
            bs = len(gt_list)
            
            # list to batch
            gt_batch = list_to_batch(gt_list, img_size=512, device=device)
            lq_batch = list_to_batch(lq_list, img_size=512, device=device)
                        
            # Train swinir:
            with accelerator.autocast():
                swinir.train(), detnet.eval()
                for p in detnet.parameters(): p.requires_grad = False
                res_batch = swinir(lq_batch)
                res_list = batch_to_list(res_batch, gt_list)
                loss_swinir_pix = F.l1_loss(input=res_batch, target=gt_batch, reduction="mean") * cfg.train.pix_weight
                _, _, feat_gt = detnet(gt_list, return_feat=True)
                _, _, feat_res = detnet(res_list, return_feat=True)
                loss_swinir_tdp = F.l1_loss(input=feat_res['features']['0'], target=feat_gt['features']['0'], reduction="mean") * 0.5 + \
                    F.l1_loss(input=feat_res['features']['1'], target=feat_gt['features']['1'], reduction="mean") * 0.5
                loss_swinir = loss_swinir_pix + loss_swinir_tdp
            opt_swinir.zero_grad()
            accelerator.backward(loss_swinir)
            opt_swinir.step(), sch_swinir.step()
            
            # Train detnet:
            with accelerator.autocast():
                swinir.eval(), detnet.train()
                for n, p in detnet.named_parameters(): 
                    if not (n in frozen_parameters): p.requires_grad = True
                with torch.no_grad():
                    res_batch = swinir(lq_batch)
                res_list = batch_to_list(res_batch, gt_list)
                cqmix_mask = F.interpolate((torch.randn(bs,1,8,8)).bernoulli_(p=0.5), scale_factor=64, mode='nearest').to(device)
                cqmix_batch = res_batch*cqmix_mask + gt_batch*(1-cqmix_mask) 
                cqmix_list = batch_to_list(cqmix_batch, gt_list)
                img_cat_list = res_list + gt_list + cqmix_list
                annot_cat_list = annot_list + annot_list + annot_list
                _, loss_dict = detnet(img_cat_list, annot_cat_list)
                loss_detnet = sum(loss_value for loss_value in loss_dict.values())
            opt_detnet.zero_grad()
            accelerator.backward(loss_detnet)
            opt_detnet.step(), sch_detnet.step()
            
            accelerator.wait_for_everyone()

            global_step += 1
            loss_records["swinir_pix"].append(loss_swinir_pix.item())
            loss_records["swinir_tdp"].append(loss_swinir_tdp.item())
            loss_records["detnet"].append(loss_detnet.item())
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch:04d}, Step: {global_step:07d}, Pix: {loss_swinir_pix.item():.4f} TDP: {loss_swinir_tdp.item():.4f}")

            # Log training loss, learning rate
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
                    
                    writer.add_scalar("train/learning_rate_swinir", opt_swinir.param_groups[0]['lr'], global_step)
                    writer.add_scalar("train/learning_rate_detnet", opt_detnet.param_groups[0]['lr'], global_step)

            # Save checkpoint
            if global_step % cfg.train.ckpt_every == 0 or (args.debug):
                if accelerator.is_local_main_process:
                    torch.save(pure_swinir.state_dict(), f"{ckpt_dir}/swinir_{global_step:07d}.pt")
                    torch.save(pure_detnet.state_dict(), f"{ckpt_dir}/detnet_{global_step:07d}.pt")

            # Save images
            if global_step % cfg.train.image_every == 0 or global_step == 1 or (args.debug):
                with torch.no_grad():
                    N = 4
                    gt_batch = list_to_batch(gt_list, img_size=512, device=device)
                    log_gt, log_lq, log_res = gt_batch[:N], lq_batch[:N], res_batch[:N] 
                    if accelerator.is_local_main_process:
                        for tag, image in [
                            ("image/gt", log_gt), ("image/lq", log_lq), ("image/res", log_res),
                        ]:
                            grid_image = make_grid(image, nrow=4)
                            writer.add_image(tag, grid_image, global_step)
                            img_name = '{}_{:06d}.png'.format(os.path.basename(tag), global_step)
                            save_image(grid_image, os.path.join(img_dir, img_name))
                        del grid_image
                    del log_gt, log_lq, log_res
                    torch.cuda.empty_cache()

            # Evaluation
            if global_step % cfg.val.val_every == 0 or (args.debug):
                swinir.eval(), detnet.eval()
                val_psnr = []
                coco_evaluator = CocoEvaluator(coco, iou_types)
                val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch",
                                total=len(val_loader), leave=False, desc="Validation")
                for idx, (val_gt_list, val_lq_list, val_annot_list, _) in enumerate(val_loader):
                    val_gt_list = list(rearrange(val_gt, 'h w c -> c h w').contiguous().float().to(device) for val_gt in val_gt_list)
                    val_lq_list = list(rearrange(val_lq, 'h w c -> c h w').contiguous().float().to(device) for val_lq in val_lq_list)
                    val_annot_list = [{k: v.long().to(device) if isinstance(v, torch.Tensor) else v for k, v in val_t.items()} for val_t in val_annot_list]
                    assert (len(val_gt_list) == 1)
                    
                    with torch.no_grad():
                        # Restoration
                        val_lq_batch = list_to_batch(val_lq_list, img_size=512, device=device)
                        val_res_batch = swinir(val_lq_batch).contiguous()
                        val_res_list = batch_to_list(val_res_batch, val_gt_list)
                        
                        # Detection
                        val_pred_list, _ = detnet(val_res_list)
                        val_pred_list = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in val_pred_list]
                    
                        # Save images
                        if idx < 5 and accelerator.is_local_main_process:
                            val_gt_box = draw_box(val_gt_list[0], val_annot_list[0])
                            val_pred_box = draw_box(val_res_list[0], val_pred_list[0])
                            save_image(val_gt_box, os.path.join(img_dir, f"val_gt_{global_step:06d}_{idx:02d}.png"))
                            save_image(val_pred_box, os.path.join(img_dir, f"val_pred_{global_step:06d}_{idx:02d}.png"))
                    
                        # Evaluation:
                        val_gt, val_res = val_gt_list[0].unsqueeze(0), val_res_list[0].unsqueeze(0)
                        _val_psnr = calculate_psnr_pt(val_res, val_gt, crop_border=0).unsqueeze(0)
                        _val_psnr = accelerator.gather_for_metrics(_val_psnr)
                        
                        res = {annot["image_id"]: pred for annot, pred in zip(val_annot_list, val_pred_list)}
                        coco_evaluator.update(res)
                        if accelerator.is_local_main_process:
                            for v in _val_psnr:
                                val_psnr += [v.item()]
                        accelerator.wait_for_everyone()
                    val_pbar.update(1)
                val_pbar.close()
                
                coco_evaluator.synchronize_between_processes()
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
                swinir.train(), detnet.train()
            
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break
        
        pbar.close()
        epoch += 1

    # Save the last model weight
    if accelerator.is_local_main_process:
        torch.save(pure_swinir.state_dict(), f"{ckpt_dir}/swinir_last.pt")
        torch.save(pure_detnet.state_dict(), f"{ckpt_dir}/detnet_last.pt")
        Logging("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(args)
