import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
import utils.filter_warning

import torch
from copy import deepcopy
from tqdm import tqdm
from utils.common import (
    instantiate_from_config, load_network,
    calculate_psnr_pt, wavelet_reconstruction
)
from utils.classification import calculate_accuracy, prepare_environment
from utils.sampler import SpacedSampler
from model import SwinIR, ControlLDM, Diffusion
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange
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
    
    teacher_clsnet = instantiate_from_config(cfg.model.teacher_clsnet)  # clsnet trained on HQ images
    load_path = cfg.train.resume_teacher_clsnet
    teacher_clsnet = load_network(teacher_clsnet, load_path, strict=cfg.train.strict_load)
    for p in teacher_clsnet.parameters(): p.requires_grad = False  # will not be trained
    Logging(f"Load Teacher ClassificationNetwork weight from checkpoint: {load_path}, strict: {cfg.train.strict_load}")
        
    clsnet = instantiate_from_config(cfg.model.clsnet)
    if cfg.train.resume_clsnet:
        load_path = cfg.train.resume_clsnet
        clsnet = load_network(clsnet, load_path, strict=cfg.train.strict_load)
        Logging(f"Load ClassificationNetwork weight from checkpoint: {load_path}, strict: {cfg.train.strict_load}")
    else:
        Logging("Initialize ClassificationNetwork from scratch")
        
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    sampler = SpacedSampler(diffusion.betas)
    
    # setup optimizer and scheduler
    diffbir_params = list(cldm.controlnet.parameters())
    opt_diffbir = torch.optim.AdamW(
        diffbir_params, lr=cfg.train.learning_rate_diffbir
    )
    opt_clsnet = torch.optim.SGD(
        clsnet.parameters(), lr=cfg.train.learning_rate_clsnet,
        momentum=0.9, weight_decay=1e-4
    )
    sch_diffbir = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_diffbir, T_max=cfg.train.train_steps,
        eta_min=1e-7
    )
    sch_clsnet = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_clsnet, T_max=cfg.train.train_steps,
        eta_min=1e-7
    )
    
    # setup data
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True, drop_last=True
    )
    Logging(f"Training dataset contains {len(dataset):,} images from {dataset.root}")
    
    val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=cfg.val.batch_size,
        num_workers=cfg.val.num_workers,
        shuffle=False, drop_last=False
    )
    Logging(f"Validation dataset contains {len(val_dataset):,} images from {val_dataset.root}")
    
    # prepare models, training logs
    swinir.eval().to(device)
    cldm.train().to(device)
    teacher_clsnet.eval().to(device)
    clsnet.train().to(device)
    diffusion.to(device)
    swinir, cldm, teacher_clsnet, clsnet, opt_diffbir, opt_clsnet, sch_diffbir, sch_clsnet, loader, val_loader = \
        accelerator.prepare(swinir, cldm, teacher_clsnet, clsnet, opt_diffbir, opt_clsnet, sch_diffbir, sch_clsnet, loader, val_loader)
    pure_swinir = accelerator.unwrap_model(swinir)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    pure_clsnet = accelerator.unwrap_model(clsnet)
    global_step = 0
    max_steps = cfg.train.train_steps
    epoch = 0
    loss_records = dict(eps=[], ce=[], fm=[])
    loss_avg_records = deepcopy(loss_records)
    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
    
    # Training:
    Logging(f"Training for {max_steps} steps...")
    while global_step < max_steps:
        pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch", total=len(loader))
        for gt, lq, label, _ in loader:
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float().to(device)
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float().to(device)
            bs = gt.size(0)
            prompt = [cfg.train.default_prompt] * bs
            
            with torch.no_grad():
                pre_res = swinir(lq)
            
            # Train diffbir:
            with accelerator.autocast():
                cldm.train()
                
                with torch.no_grad():
                    z_0 = pure_cldm.vae_encode(gt * 2 - 1)
                    cond = pure_cldm.prepare_condition(pre_res, prompt)
                    t = torch.randint(0, diffusion.num_timesteps, (bs,), device=device)
                loss_eps = diffusion.p_losses(cldm, z_0, t, cond)
            
            opt_diffbir.zero_grad()
            accelerator.backward(loss_eps)
            opt_diffbir.step(), sch_diffbir.step()
            
            # Train clsnet:
            with accelerator.autocast():
                cldm.eval()
                bs2 = bs//2  # For training clsnet, use 1/2 batch from DiffBIR and 1/2 from HQ images
                # NOTE: This is the same setting as done in EDTR training
                
                # restoration
                with torch.no_grad():
                    cond = dict(c_txt=cond["c_txt"][:bs2], c_img=cond["c_img"][:bs2])
                    
                    z = sampler.sample(
                        model=cldm, device=device, steps=50, batch_size=bs2, x_size=z_0.shape[1:],
                        cond=cond, uncond=None, cfg_scale=1.0, x_T=None,
                        progress=accelerator.is_local_main_process, progress_leave=False
                    )
                    res = wavelet_reconstruction((pure_cldm.vae_decode(z) + 1) / 2, pre_res[:bs2])
            
                # task loss
                pred, feat_student = clsnet(torch.cat((res, gt[bs2:]), dim=0), return_feat=True)
                loss_ce = F.cross_entropy(pred, label, reduction="mean", label_smoothing=0.0) * cfg.train.weight_ce
                
                # feature-matching loss
                with torch.no_grad():
                    _, feat_teacher = teacher_clsnet(gt, return_feat=True)
                loss_fm = F.l1_loss(input=feat_student, target=feat_teacher, reduction="mean") * cfg.train.weight_fm
                
            opt_clsnet.zero_grad()
            accelerator.backward(loss_ce + loss_fm)
            opt_clsnet.step(), sch_clsnet.step()

            accelerator.wait_for_everyone()

            global_step += 1
            loss_records["eps"].append(loss_eps.item())
            loss_records["ce"].append(loss_ce.item())
            loss_records["fm"].append(loss_fm.item())
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch:04d}, Steps: {global_step:07d}, Eps: {loss_eps.item():.3f}, CE: {loss_ce.item():.3f}, FM: {loss_fm.item():.3f}")

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
                    writer.add_scalar("train/learning_rate_clsnet", opt_clsnet.param_groups[0]['lr'], global_step)

            # save checkpoint
            if global_step % cfg.train.ckpt_every == 0 or (args.debug):
                if accelerator.is_local_main_process:
                    torch.save(pure_cldm.controlnet.state_dict(), f"{ckpt_dir}/cldm_{global_step:07d}.pt")
                    torch.save(pure_clsnet.state_dict(), f"{ckpt_dir}/clsnet_{global_step:07d}.pt")

            # save images
            if global_step % cfg.train.image_every == 0 or global_step == 1 or (args.debug):
                with torch.no_grad():
                    N = 4
                    log_gt, log_lq, log_pre_res = gt[:N], lq[:N], pre_res[:N]
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
            if global_step % cfg.val.val_every == 0 and global_step > 0 or (args.debug):
                cldm.eval(), clsnet.eval()
                val_psnr, val_acc1 = [], []
                val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch",
                                total=len(val_loader), leave=False, desc="Validation")
                for val_gt, val_lq, val_label, _ in val_loader:
                    val_gt = rearrange(val_gt, "b h w c -> b c h w").contiguous().float().to(device)
                    val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float().to(device)
                    val_bs = val_gt.size(0)
                    val_prompt = [cfg.val.default_prompt] * val_bs
                    
                    with torch.no_grad():
                        # Restoration
                        val_pre_res = swinir(val_lq)
                        val_cond = pure_cldm.prepare_condition(val_pre_res, val_prompt)    
                        val_z = sampler.sample(
                            model=cldm, device=device, steps=50, batch_size=val_bs, x_size=z_0.shape[1:],
                            cond=val_cond, uncond=None, cfg_scale=1.0, x_T=None,
                            progress=accelerator.is_local_main_process, progress_leave=False
                        )
                        val_res = wavelet_reconstruction((pure_cldm.vae_decode(val_z) + 1) / 2, val_pre_res)
                        
                        # Clasification
                        val_pred = clsnet(val_res)
                        
                        # calculate metrics
                        val_gt, val_res, val_pred, val_label = accelerator.gather_for_metrics((val_gt, val_res, val_pred, val_label))
                        if accelerator.is_local_main_process:
                            val_psnr += calculate_psnr_pt(val_res, val_gt, crop_border=0).tolist()
                            val_acc1 += [calculate_accuracy(val_pred, val_label, topk=(1, 5))[0]] * val_gt.size(0)
                        accelerator.wait_for_everyone()
                    val_pbar.update(1)
                val_pbar.close()
                
                if accelerator.is_local_main_process:
                    avg_val_psnr = torch.tensor(val_psnr).mean().item()
                    avg_val_acc1 = torch.tensor(val_acc1).mean().item()
                    for tag, val in [
                        ("val/psnr", avg_val_psnr),
                        ("val/acc1", avg_val_acc1),
                    ]:
                        Logging(f"{tag}: {val:.4f}")
                        writer.add_scalar(tag, val, global_step)
                
                cldm.train(), clsnet.train()
                del val_gt, val_lq, val_label, val_pre_res, val_cond, val_z, val_res, val_pred
                torch.cuda.empty_cache()
            
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break
        
        pbar.close()
        epoch += 1

    # save the last model weight
    if accelerator.is_local_main_process:
        torch.save(pure_swinir.state_dict(), f"{ckpt_dir}/swinir_last.pt")
        torch.save(pure_cldm.controlnet.state_dict(), f"{ckpt_dir}/cldm_last.pt")
        torch.save(pure_clsnet.state_dict(), f"{ckpt_dir}/clsnet_last.pt")
        Logging("Done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(args)
