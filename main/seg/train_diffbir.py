import os
import sys
import math
import torch
import safetensors
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from copy import deepcopy
from tqdm import tqdm
from model import SwinIR, ControlLDM, Diffusion
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.common import (
    instantiate_from_config, wavelet_reconstruction, load_network,
    calculate_psnr_pt, set_logger, copy_opt_file, print_attn_type
)
from utils.sampler import SpacedSampler
from utils.segmentation import convert2color, calculate_mat, compute_iou
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
    # if args.debug: cfg.train.batch_size, cfg.val.batch_size = 4, 1
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
        
    teacher_segnet = instantiate_from_config(cfg.model.teacher_segnet)
    load_path = cfg.train.resume_teacher_segnet
    teacher_segnet = load_network(teacher_segnet, load_path, strict=cfg.train.strict_load)
    for p in teacher_segnet.parameters(): p.requires_grad = False  # will not be trained
    Logging(f"Load Teacher SegmentationNetwork weight from checkpoint: {load_path}, strict: {cfg.train.strict_load}")
    
    segnet = instantiate_from_config(cfg.model.segnet)
    if cfg.train.resume_segnet:
        segnet = load_network(segnet, cfg.train.resume_segnet, strict=cfg.train.strict_load)
        Logging(f"Load SegmentationNetwork weight from checkpoint: {cfg.train.resume_segnet}, strict: {cfg.train.strict_load}")
    else:
        Logging("Initialize SegmentationNetwork from scratch")
    
    # Setup optimizer:
    diffbir_params = list(cldm.controlnet.parameters())
    opt_diffbir = torch.optim.AdamW(
        diffbir_params, lr=cfg.train.learning_rate_diffbir
    )
    
    opt_segnet = torch.optim.SGD(
        segnet.parameters(), lr=cfg.train.learning_rate_segnet,
        momentum=0.9, weight_decay=1e-4
    )
    
    # Setup scheduler:
    sch_diffbir = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_diffbir, T_max=cfg.train.train_steps,
        eta_min=1e-7
    )
    
    sch_segnet = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_segnet, T_max=cfg.train.train_steps,
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
    
    if cfg.val.batch_size == -1:
        num_gpus = accelerator.state.num_processes
        cfg.val.batch_size = num_gpus
        Logging(f"Validation batch size is automatically set to the number of available GPUs: {num_gpus}")
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

    # Prepare models for training
    swinir.eval().to(device)
    cldm.train().to(device)
    teacher_segnet.eval().to(device)
    segnet.train().to(device)
    diffusion.to(device)
    swinir, cldm, teacher_segnet, segnet, opt_diffbir, opt_segnet, sch_diffbir, sch_segnet, loader, val_loader = \
        accelerator.prepare(swinir, cldm, teacher_segnet, segnet, opt_diffbir, opt_segnet, sch_diffbir, sch_segnet, loader, val_loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    pure_segnet = accelerator.unwrap_model(segnet)
    
    # Define variables related to training
    global_step = 0
    max_steps = cfg.train.train_steps
    epoch = 0
    loss_records = dict(eps=[], ce=[], fm=[])
    loss_avg_records = deepcopy(loss_records)
    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
        
    # Training
    if accelerator.mixed_precision == 'fp16':
        Logging("Mixed precision is applied")
    Logging(f"Training for {max_steps} steps...")
    
    while global_step < max_steps:
        pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch", total=len(loader))
        for gt, lq, mask, _ in loader:
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float().to(device)
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float().to(device)
            mask = mask.long()
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
            
            # Train segnet:
            with accelerator.autocast():
                cldm.eval(), segnet.train()
                for p in segnet.parameters(): p.requires_grad = True
                bs2 = bs//2  # For training segnet, use 1/2 batch from DiffBIR and 1/2 from HQ images
                # NOTE: This is the same setting as done in EDTR training
                
                # Restoration
                with torch.no_grad():                    
                    cond = dict(c_txt=cond["c_txt"][:bs2], c_img=cond["c_img"][:bs2])
                    
                    z = sampler.sample(
                        model=cldm, device=device, steps=50, batch_size=bs2, x_size=z_0.shape[1:],
                        cond=cond, uncond=None, cfg_scale=1.0, x_T=None,
                        progress=accelerator.is_local_main_process, progress_leave=False
                    )
                    res = wavelet_reconstruction((pure_cldm.vae_decode(z) + 1) / 2, pre_res[:bs2])
                
                # Task loss
                pred, feat_student = segnet(torch.cat((res, gt[bs2:]), dim=0), return_feat=True)
                loss_ce = F.cross_entropy(pred["out"], mask, ignore_index=255) * cfg.train.weight_ce
                
                # Feature-Matching loss
                with torch.no_grad():
                    _, feat_teacher = teacher_segnet(gt, return_feat=True)
                loss_fm = F.l1_loss(input=feat_student["C5"], target=feat_teacher["C5"], reduction="mean") * cfg.train.weight_fm
                
            opt_segnet.zero_grad()
            accelerator.backward(loss_ce + loss_fm)
            opt_segnet.step(), sch_segnet.step()

            accelerator.wait_for_everyone()

            global_step += 1
            loss_records["eps"].append(loss_eps.item())
            loss_records["ce"].append(loss_ce.item())
            loss_records["fm"].append(loss_fm.item())
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch:04d}, Steps: {global_step:07d}, Eps: {loss_eps.item():.3f}, CE: {loss_ce.item():.3f}, FM: {loss_fm.item():.3f}")

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
                    
                    writer.add_scalar("train/learning_rate_diffbir", opt_diffbir.param_groups[0]['lr'], global_step)
                    writer.add_scalar("train/learning_rate_segnet", opt_segnet.param_groups[0]['lr'], global_step)

            # Save checkpoint
            if global_step % cfg.train.ckpt_every == 0 or (args.debug):
                if accelerator.is_local_main_process:
                    torch.save(pure_cldm.controlnet.state_dict(), f"{ckpt_dir}/cldm_{global_step:07d}.pt")
                    torch.save(pure_cldm.vae.decoder.state_dict(), f"{ckpt_dir}/decoder_{global_step:07d}.pt")
                    torch.save(pure_segnet.state_dict(), f"{ckpt_dir}/segnet_{global_step:07d}.pt")

            # Save images
            if global_step % cfg.train.image_every == 0 or global_step == 1 or (args.debug):
                with torch.no_grad():
                    N = 2
                    log_gt, log_lq, log_pre_res = gt[:N], lq[:N], pre_res[:N]
                    log_cond = {k:v[:N] for k, v in cond.items()}
                    cldm.eval()
                    z = sampler.sample(
                        model=cldm, device=device, steps=50, batch_size=len(log_gt), x_size=z_0.shape[1:],
                        cond=log_cond, uncond=None, cfg_scale=1.0, x_T=None,
                        progress=accelerator.is_local_main_process, progress_leave=False
                    )
                    log_res = wavelet_reconstruction((pure_cldm.vae_decode(z) + 1) / 2, log_pre_res)
                    log_mask, log_pred = convert2color(mask[:N]), convert2color(pred["out"][:N].argmax(1))
                    
                    if accelerator.is_local_main_process:
                        for tag, image in [
                            ("image/gt", log_gt), ("image/lq", log_lq),
                            ("image/pre_restored", log_pre_res), ("image/restored", log_res),
                            ("image/mask", log_mask), ("image/pred", log_pred),  
                        ]:
                            grid_image = make_grid(image, nrow=4)
                            writer.add_image(tag, grid_image, global_step)
                            img_name = '{}_{:06d}.png'.format(os.path.basename(tag), global_step)
                            save_image(grid_image, os.path.join(img_dir, img_name))
                        del grid_image
                    del log_gt, log_lq, log_pre_res, log_res, log_mask, log_pred
                    torch.cuda.empty_cache()
                    cldm.train()
            
            # Evaluation
            if global_step % cfg.val.val_every == 0 and global_step > 0 or (args.debug):
                cldm.eval(), segnet.eval()
                n_classes = 21
                confmat = torch.zeros((n_classes, n_classes), device=device)
                val_psnr = []
                val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch",
                                total=len(val_loader), leave=False, desc="Validation")
                
                for val_gt, val_lq, val_mask, _ in val_loader:
                    val_gt = rearrange(val_gt, "b h w c -> b c h w").contiguous().float().to(device)
                    val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float().to(device)
                    val_mask = val_mask.long()
                    val_bs = val_gt.size(0)
                    val_prompt = [cfg.val.default_prompt] * val_bs
                    
                    with torch.no_grad():
                        # Padding
                        h, w = val_lq.shape[2:]
                        ph, pw = math.ceil(h/64)*64 - h, math.ceil(w/64)*64 - w
                        val_lq_padded = F.pad(val_lq, pad=(0, pw, 0, ph), mode='replicate')
                        
                        # Restoration
                        val_pre_res = swinir(val_lq_padded)
                        val_cond = pure_cldm.prepare_condition(val_pre_res, val_prompt)
                        val_z = sampler.sample(
                            model=cldm, device=device, steps=50, batch_size=val_bs, x_size=val_cond["c_img"].shape[1:],
                            cond=val_cond, uncond=None, cfg_scale=1.0, x_T=None,
                            progress=accelerator.is_local_main_process, progress_leave=False
                        )
                        val_res = wavelet_reconstruction((pure_cldm.vae_decode(val_z) + 1) / 2, val_pre_res)[:,:,:h,:w]
                        
                        # Segmentation
                        val_pred = segnet(val_res)
                        
                        # Calculate metrics
                        val_psnr_batch = calculate_psnr_pt(val_res, val_gt, crop_border=0).unsqueeze(0)
                        val_mat = calculate_mat(val_mask.flatten(), val_pred['out'].argmax(1).flatten(), n=n_classes).unsqueeze(0)
                        val_psnr_batch, val_mat = accelerator.gather_for_metrics((val_psnr_batch, val_mat))
                        if accelerator.is_local_main_process:
                            for v in val_psnr_batch: val_psnr += [v.item()]
                            confmat += val_mat.sum(0)
                        
                        accelerator.wait_for_everyone()
                    val_pbar.update(1)
                val_pbar.close()
                
                if accelerator.is_local_main_process:
                    miou = compute_iou(confmat).mean().item() * 100
                    avg_val_psnr = torch.tensor(val_psnr).mean().item()
                    for tag, val in [
                        ("val/miou", miou),
                        ("val/psnr", avg_val_psnr),
                    ]:
                        Logging(f"{tag}: {val:.4f}")
                        writer.add_scalar(tag, val, global_step)
                cldm.train(), segnet.train()
                del val_gt, val_lq, val_mask, val_pre_res, val_cond, val_z, val_res, val_pred
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
        torch.save(pure_segnet.state_dict(), f"{ckpt_dir}/segnet_last.pt")
        Logging("Done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(args)
