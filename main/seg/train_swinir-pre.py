import os
import sys
import math
import torch
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from tqdm import tqdm
from model import SwinIR
from einops import rearrange
from omegaconf import OmegaConf
from accelerate import Accelerator
from argparse import ArgumentParser
from torch.nn import functional as F
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from utils.common import (
    instantiate_from_config, calculate_psnr_pt,
    copy_opt_file, set_logger, print_attn_type
)


def main(args) -> None:
    # Setup accelerator
    cfg = OmegaConf.load(args.config)
    # if args.debug: cfg.train.batch_size, cfg.val.batch_size = 8, 2
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
    if cfg.train.resume_swinir:
        swinir.load_state_dict(torch.load(cfg.train.resume_swinir, map_location="cpu"), strict=True)
        Logging(f"Load SwinIR weight from checkpoint: {cfg.train.resume_swinir}")
    else:
        Logging("Initialize SwinIR from scratch")
    
    # Setup optimizer:
    opt = torch.optim.AdamW(
        swinir.parameters(), lr=cfg.train.learning_rate,
        weight_decay=0
    )
    
    # Setup scheduler:
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.train.train_steps,
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

    # Prepare models for training
    swinir.train().to(device)
    swinir, opt, sch, loader, val_loader = accelerator.prepare(swinir, opt, sch, loader, val_loader)
    pure_swinir = accelerator.unwrap_model(swinir)

    # Define variables related to training
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    epoch = 0
    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
    
    # Training
    if accelerator.mixed_precision == 'fp16':
        Logging("Mixed precision is applied")
    Logging(f"Training for {max_steps} steps...")
    
    while global_step < max_steps:
        pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch", total=len(loader))
        for gt, lq, _, _ in loader:
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float().to(device)
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float().to(device)
            
            with accelerator.autocast():
                res = swinir(lq)
                loss = F.l1_loss(input=res, target=gt, reduction="mean") * 255.0
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step(), sch.step()
            
            accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(loss.item())
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch:04d}, Steps: {global_step:07d}, Loss: {loss.item():.6f}")

            # Log training loss, learning rate
            if global_step % cfg.train.log_every == 0 or (args.debug):
                avg_loss = accelerator.gather(torch.tensor(step_loss, device=device).unsqueeze(0)).mean().item()
                step_loss.clear()
                loss_summary = f"[{global_step:05d}/{max_steps:05d}] Training loss: {avg_loss:.4f}"
                Logging(loss_summary, print=False)
                if accelerator.is_local_main_process:
                    writer.add_scalar("train/loss_step", avg_loss, global_step)
                    writer.add_scalar("train/learning_rate", opt.param_groups[0]['lr'], global_step)

            # Save checkpoint
            if global_step % cfg.train.ckpt_every == 0 or (args.debug):
                if accelerator.is_local_main_process:
                    torch.save(pure_swinir.state_dict(), f"{ckpt_dir}/{global_step:07d}.pt")

            # Save images
            if global_step % cfg.train.image_every == 0 or global_step == 1 or (args.debug):
                swinir.eval()
                N = 12
                log_gt, log_lq = gt[:N], lq[:N]
                with torch.no_grad():
                    log_res = swinir(log_lq)
                if accelerator.is_local_main_process:
                    for tag, image in [
                        ("image/restored", log_res),
                        ("image/gt", log_gt),
                        ("image/lq", log_lq),
                    ]:
                        grid_image = make_grid(image, nrow=4)
                        writer.add_image(tag, grid_image, global_step)
                        img_name = '{}_{:06d}.png'.format(os.path.basename(tag), global_step)
                        save_image(grid_image, os.path.join(img_dir, img_name))
                swinir.train()

            # Evaluation
            if global_step % cfg.val.val_every == 0 or (args.debug):
                swinir.eval()
                val_psnr = []
                val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch",
                                total=len(val_loader), leave=False, desc="Validation")
                for val_gt, val_lq, _, _ in val_loader:
                    val_gt = rearrange(val_gt, "b h w c -> b c h w").contiguous().float().to(device)
                    val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float().to(device)
                    
                    h, w = val_lq.shape[2:]
                    ph, pw = math.ceil(h/64)*64 - h, math.ceil(w/64)*64 - w
                    val_lq_padded = F.pad(val_lq, pad=(0, pw, 0, ph), mode='replicate')
                    
                    with torch.no_grad():
                        val_res = swinir(val_lq_padded)[:,:,:h,:w]
                        _val_psnr = calculate_psnr_pt(val_res, val_gt, crop_border=0).unsqueeze(0)
                        _val_psnr = accelerator.gather_for_metrics(_val_psnr)
                        
                        if accelerator.is_local_main_process:
                            for v in _val_psnr:
                                val_psnr += [v.item()]
                        accelerator.wait_for_everyone()
                    val_pbar.update(1)
                val_pbar.close()
                
                if accelerator.is_local_main_process:
                    avg_val_psnr = torch.tensor(val_psnr).mean().item()
                    for tag, val in [
                        ("val/psnr", avg_val_psnr)
                    ]:
                        Logging(f"{tag}: {val:.4f}")
                        writer.add_scalar(tag, val, global_step)
                swinir.train()
            
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break
        
        pbar.close()
        epoch += 1

    # Save the last model weight
    if accelerator.is_local_main_process:
        torch.save(pure_swinir.state_dict(), f"{ckpt_dir}/swinir_last.pt")
        Logging("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(args)
