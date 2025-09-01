import os
import sys
import math
import torch
import random
import safetensors
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from copy import deepcopy
from tqdm import tqdm
from utils.common import (
    instantiate_from_config, wavelet_reconstruction, load_network, set_logger,
    calculate_psnr_pt, copy_opt_file, print_attn_type
)
from utils.classification import calculate_accuracy
from utils.sampler import SpacedSampler
from model import SwinIR, ControlLDM, Diffusion
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange
from argparse import ArgumentParser
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision.utils import make_grid, save_image


def main(args) -> None:
    # Prevent memory fragmentation:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Setup accelerator
    cfg = OmegaConf.load(args.config)
    if args.debug: cfg.train.batch_size, cfg.val.batch_size = 8, 8
    accelerator = Accelerator(split_batches=True, mixed_precision=cfg.train.precision)
    set_seed(cfg.train.seed)
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

    # Create models
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    swinir.load_state_dict(torch.load(cfg.train.resume_swinir, map_location="cpu"), strict=True)
    Logging(f"Load SwinIR weight from checkpoint: {cfg.train.resume_swinir}")
    if not cfg.model.pre_restoration:
        Logging("NOT using pre-restoration")

    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    if 'turbo' in cfg.train.sd_path:
        sd = safetensors.torch.load_file(cfg.train.sd_path, device="cpu")
        unused = cldm.load_pretrained_sd(sd, is_turbo=True)
        Logging(f"Stable Diffusion **TURBO** is using")    
    else:
        sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
        unused = cldm.load_pretrained_sd(sd)
    Logging(f"Load pretrained SD weight from {cfg.train.sd_path}")
    Logging(f"Unused weights: {unused}", print=False)
        
    if cfg.train.resume_cldm:
        cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume_cldm, map_location="cpu"))
        Logging(f"Load ControlNet weight from checkpoint: {cfg.train.resume_cldm}")
    else:
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
        Logging(f"Load ControlNet weight from pretrained SD")
        Logging(f"Weights initialized with newly added zeros: {init_with_new_zero}", print=False)
        Logging(f"Weights initialized from scratch: {init_with_scratch}", print=False)
    
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
    
    # Setup optimizer:
    edtr_params = list(cldm.controlnet.parameters())
    if cldm.vae.train_decoder:
        edtr_params += list(cldm.vae.decoder.parameters())
    else:
        Logging("Decoder in EDTR is NOT trainable")
    opt_edtr = torch.optim.AdamW(
        edtr_params, lr=cfg.train.learning_rate_edtr
    )
    
    opt_clsnet = torch.optim.SGD(
        clsnet.parameters(), lr=cfg.train.learning_rate_clsnet,
        momentum=0.9, weight_decay=1e-4
    )
    
    # Setup scheduler:
    sch_edtr = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_edtr, T_max=cfg.train.train_steps,
        eta_min=1e-7
    )
    
    sch_clsnet = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_clsnet, T_max=cfg.train.train_steps,
        eta_min=1e-7
    )
    
    # Setup data
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
    
    # Diffusion functions
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    sampler = SpacedSampler(diffusion.betas)
    tp, N = cfg.train.start_timestep, cfg.train.num_timesteps
    used_timesteps = [math.floor(tp/N*i) for i in range(1, N+1)]
    Logging(f"Used timesteps are specified as {used_timesteps}, total number of {N}")
    
    # Prepare models for training
    swinir.eval().to(device)
    cldm.train().to(device)
    teacher_clsnet.eval().to(device)
    clsnet.train().to(device)
    diffusion.to(device)
    swinir, cldm, teacher_clsnet, clsnet, opt_edtr, opt_clsnet, sch_edtr, sch_clsnet, loader, val_loader = \
        accelerator.prepare(swinir, cldm, teacher_clsnet, clsnet, opt_edtr, opt_clsnet, sch_edtr, sch_clsnet, loader, val_loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    pure_clsnet = accelerator.unwrap_model(clsnet)
    
    # Define variables related to training
    global_step = 0
    max_steps = cfg.train.train_steps
    epoch = 0
    loss_records = dict(hlf=[], ce=[], fm=[])
    loss_avg_records = deepcopy(loss_records)
    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
    
    # Training
    if accelerator.mixed_precision == 'fp16':
        Logging("Mixed precision is applied")
    Logging(f"Training for {max_steps} steps...")
    
    while global_step < max_steps:
        pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch", total=len(loader))
        for gt, lq, label, _ in loader:
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float().to(device)
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float().to(device)
            bs = gt.size(0)
            prompt = [cfg.train.default_prompt] * bs
            
            # Pre-restoration
            pre_res = lq
            if cfg.model.pre_restoration:
                with torch.no_grad(): pre_res = swinir(lq)
            
            # Train edtr:
            with accelerator.autocast():
                cldm.train(), clsnet.eval()
                for p in clsnet.parameters(): p.requires_grad = False
                bs1 = bs//2  # For classification, we only use 1/2 batch for training EDTR due to memory issue
                
                # Prepare condition
                with torch.no_grad():
                    z_pre_res = pure_cldm.vae_encode(pre_res[:bs1] * 2 - 1, sample=False)
                    t = torch.tensor([random.choice(used_timesteps) for _ in range(bs1)], dtype=torch.int64).to(device)
                    cond = dict(c_txt=pure_cldm.clip.encode(prompt[:bs1]), c_img=z_pre_res[:bs1])
                
                # Partial diffusion and one-step denoising
                cldm_outputs = diffusion.reverse(cldm, t, z_pre_res, cond, noise=None)
                res = wavelet_reconstruction((pure_cldm.vae_decode(cldm_outputs['x_pred']) + 1) / 2, pre_res[:bs1])
                
                # Calculate HLF loss
                _, feat_gt = clsnet(gt[:bs1], return_feat=True)
                _, feat_res = clsnet(res, return_feat=True)
                with torch.no_grad():
                    _, teacher_feat_gt = teacher_clsnet(gt[:bs1], return_feat=True)
                _, teacher_feat_res = teacher_clsnet(res, return_feat=True)
                loss_hlf = (F.l1_loss(input=teacher_feat_res, target=teacher_feat_gt, reduction="mean") + \
                    F.l1_loss(input=feat_res, target=feat_gt, reduction="mean")) * cfg.train.weight_hlf
            
            opt_edtr.zero_grad()
            accelerator.backward(loss_hlf)
            opt_edtr.step(), sch_edtr.step()
            
            # Train clsnet:
            with accelerator.autocast():
                cldm.eval(), clsnet.train()
                for p in clsnet.parameters(): p.requires_grad = True
                bs2 = bs//2  # For training clsnet, use 1/2 batch from EDTR and 1/2 from HQ images
                
                with torch.no_grad():
                    # Prepare condition
                    assert (bs1 >= bs2)
                    cond = dict(c_txt=cond["c_txt"][:bs2], c_img=cond["c_img"][:bs2])
                    
                    # Partial diffusion
                    noise = torch.randn_like(z_pre_res[:bs2])
                    t = torch.tensor([tp] * bs2, dtype=torch.int64).to(device)
                    z_partial = diffusion.q_sample(x_start=z_pre_res[:bs2], t=t, noise=noise) 
                    
                    # Short-step denoising
                    z = sampler.manual_sample_with_timesteps(
                        model=cldm, device=device, x_T=z_partial, steps=len(used_timesteps), used_timesteps=used_timesteps,
                        batch_size=bs2, x_size=z_pre_res.shape[1:], cond=cond, uncond=None, cfg_scale=1.0, 
                        progress=accelerator.is_local_main_process, progress_leave=False
                    )
                    res = wavelet_reconstruction((pure_cldm.vae_decode(z) + 1) / 2, pre_res[:bs2])
                
                # Task loss
                pred, feat_student = clsnet(torch.cat((res, gt[bs2:]), dim=0), return_feat=True)
                loss_ce = F.cross_entropy(pred, label, reduction="mean", label_smoothing=0.0) * cfg.train.weight_ce
                
                # Feature-Matching loss
                with torch.no_grad():
                    _, feat_teacher = teacher_clsnet(gt, return_feat=True)
                loss_fm = F.l1_loss(input=feat_student, target=feat_teacher, reduction="mean") * cfg.train.weight_fm
                
            opt_clsnet.zero_grad()
            accelerator.backward(loss_ce + loss_fm)
            opt_clsnet.step(), sch_clsnet.step()

            accelerator.wait_for_everyone()

            global_step += 1
            loss_records["hlf"].append(loss_hlf.item())
            loss_records["ce"].append(loss_ce.item())
            loss_records["fm"].append(loss_fm.item())
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch:04d}, Steps: {global_step:07d}, HLF: {loss_hlf.item():.3f}, CE: {loss_ce.item():.3f}, FM: {loss_fm.item():.3f}")

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
                    
                    writer.add_scalar("train/learning_rate_edtr", opt_edtr.param_groups[0]['lr'], global_step)
                    writer.add_scalar("train/learning_rate_clsnet", opt_clsnet.param_groups[0]['lr'], global_step)

            # Save checkpoint
            if global_step % cfg.train.ckpt_every == 0 or (args.debug):
                if accelerator.is_local_main_process:
                    torch.save(pure_cldm.controlnet.state_dict(), f"{ckpt_dir}/cldm_{global_step:07d}.pt")
                    torch.save(pure_cldm.vae.decoder.state_dict(), f"{ckpt_dir}/decoder_{global_step:07d}.pt")
                    torch.save(pure_clsnet.state_dict(), f"{ckpt_dir}/clsnet_{global_step:07d}.pt")

            # Save images
            if global_step % cfg.train.image_every == 0 or global_step == 1 or (args.debug):
                with torch.no_grad():
                    N = 4
                    log_gt, log_lq, log_pre_res, log_res = gt[:N], lq[:N], pre_res[:N], res[:N]
                    
                    if accelerator.is_local_main_process:
                        for tag, image in [
                            ("image/gt", log_gt), ("image/lq", log_lq),
                            ("image/pre_restored", log_pre_res), ("image/restored_final", log_res),
                        ]:
                            grid_image = make_grid(image, nrow=4)
                            writer.add_image(tag, grid_image, global_step)
                            img_name = '{:06d}_{}.png'.format(global_step, os.path.basename(tag))
                            save_image(grid_image, os.path.join(img_dir, img_name))
                        del grid_image
                    del log_gt, log_lq, log_pre_res, log_res
                    torch.cuda.empty_cache()
            
            # Evaluation
            if global_step % cfg.val.val_every == 0 and global_step > 0 or (args.debug):
                cldm.eval(), clsnet.eval()
                val_psnr, val_acc1 = [], []
                val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch",
                                total=len(val_loader), leave=False, desc="Validation")
                
                val_tp, val_N = cfg.val.start_timestep, cfg.val.num_timesteps
                val_used_timesteps = [math.floor(val_tp/val_N*i) for i in range(1, val_N+1)]
                Logging(f"Used val timesteps are specified as {val_used_timesteps}, total number of {val_N}")
                
                for val_gt, val_lq, val_label, _ in val_loader:
                    val_gt = rearrange(val_gt, "b h w c -> b c h w").contiguous().float().to(device)
                    val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float().to(device)
                    val_bs = val_gt.size(0)
                    val_prompt = [cfg.val.default_prompt] * val_bs
                    
                    with torch.no_grad():
                        # Pre-restoration
                        val_pre_res = val_lq
                        if cfg.model.pre_restoration: val_pre_res = swinir(val_lq)
                        
                        # Prepare condition
                        val_z_pre_res = pure_cldm.vae_encode(val_pre_res * 2 - 1, sample=False)
                        val_cond = dict(c_txt=pure_cldm.clip.encode(val_prompt), c_img=val_z_pre_res)
                        
                        # Partial diffusion
                        val_noise = torch.randn_like(val_z_pre_res)
                        val_t = torch.tensor([val_tp] * val_bs, dtype=torch.int64).to(device)
                        val_z_partial = diffusion.q_sample(x_start=val_z_pre_res, t=val_t, noise=val_noise)
                        
                        # Short-step denoising
                        val_z = sampler.manual_sample_with_timesteps(
                            model=cldm, device=device, x_T=val_z_partial, steps=len(val_used_timesteps), used_timesteps=val_used_timesteps,
                            batch_size=val_bs, x_size=val_z_pre_res.shape[1:], cond=val_cond, uncond=None, cfg_scale=1.0, 
                            progress=accelerator.is_local_main_process, progress_leave=False
                        )
                        val_res = wavelet_reconstruction((pure_cldm.vae_decode(val_z) + 1) / 2, val_pre_res)
                        
                        # Clasification
                        val_pred = clsnet(val_res, normalize=True)
                        
                        # Calculate metrics
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
                del val_gt, val_lq, val_label, val_pre_res, val_z_pre_res, val_cond, val_t, val_z_partial, val_res, val_pred
                torch.cuda.empty_cache()
            
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break
        
        pbar.close()
        epoch += 1

    # Save the last model weight
    if accelerator.is_local_main_process:
        torch.save(pure_cldm.controlnet.state_dict(), f"{ckpt_dir}/cldm_last.pt")
        torch.save(pure_cldm.vae.decoder.state_dict(), f"{ckpt_dir}/decoder_last.pt")
        torch.save(pure_clsnet.state_dict(), f"{ckpt_dir}/clsnet_last.pt")
        Logging("Done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(args)
