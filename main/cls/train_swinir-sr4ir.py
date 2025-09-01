import os
import sys
import torch
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from copy import deepcopy
from tqdm import tqdm
from utils.common import (
    instantiate_from_config, load_network, set_logger,
    calculate_psnr_pt, copy_opt_file
)
from utils.classification import calculate_accuracy
from model import SwinIR
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
    # Setup accelerator
    cfg = OmegaConf.load(args.config)
    accelerator = Accelerator(split_batches=True, mixed_precision=cfg.train.precision)
    set_seed(231)
    device = accelerator.device
    
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
        Logging(f"Random seed: {cfg.train.seed}")

    # Create models
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    if cfg.train.resume_swinir:
        swinir.load_state_dict(torch.load(cfg.train.resume_swinir, map_location="cpu"), strict=True)
        Logging(f"Load SwinIR weight from checkpoint: {cfg.train.resume_swinir}")
    else:
        Logging("Initialize SwinIR from scratch")
    
    clsnet = instantiate_from_config(cfg.model.clsnet)
    if cfg.train.resume_clsnet:
        load_path = cfg.train.resume_clsnet
        clsnet = load_network(clsnet, load_path, strict=cfg.train.strict_load)
        Logging(f"Load ClassificationNetwork weight from checkpoint: {load_path}, strict: {cfg.train.strict_load}")
    else:
        Logging("Initialize ClassificationNetwork from scratch")
    
    # Setup optimizer:
    opt_swinir = torch.optim.AdamW(
        swinir.parameters(), lr=cfg.train.learning_rate_swinir,
        weight_decay=0
    )
    
    opt_clsnet = torch.optim.SGD(
        clsnet.parameters(), lr=cfg.train.learning_rate_clsnet,
        momentum=0.9, weight_decay=1e-4
    )
    
    # Setup scheduler:
    sch_swinir = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_swinir, T_max=cfg.train.train_steps,
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

    # Prepare models for training
    swinir.train().to(device)
    clsnet.train().to(device)
    swinir, clsnet, opt_swinir, opt_clsnet, sch_swinir, sch_clsnet, loader, val_loader = \
        accelerator.prepare(swinir, clsnet, opt_swinir, opt_clsnet, sch_swinir, sch_clsnet, loader, val_loader)
    pure_swinir = accelerator.unwrap_model(swinir)
    pure_clsnet = accelerator.unwrap_model(clsnet)

    # Define variables related to training
    global_step = 0
    max_steps = cfg.train.train_steps
    epoch = 0
    loss_records = dict(swinir_pix=[], swinir_tdp=[], clsnet=[])
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

            # Train swinir:
            with accelerator.autocast():
                swinir.train(), clsnet.eval()
                for p in clsnet.parameters(): p.requires_grad = False
                res = swinir(lq)
                loss_swinir_pix = F.l1_loss(input=res, target=gt, reduction="mean") * cfg.train.pix_weight
                _, feat_gt = clsnet(gt, normalize=True, return_feat=True)
                _, feat_res = clsnet(res, normalize=True, return_feat=True)
                loss_swinir_tdp = F.l1_loss(input=feat_res, target=feat_gt, reduction="mean")
                loss_swinir = loss_swinir_pix + loss_swinir_tdp
            opt_swinir.zero_grad()
            accelerator.backward(loss_swinir)
            opt_swinir.step(), sch_swinir.step()
            
            # Train clsnet:
            with accelerator.autocast():
                swinir.eval(), clsnet.train()
                for p in clsnet.parameters(): p.requires_grad = True
                with torch.no_grad():
                    res = swinir(lq)
                mask = F.interpolate((torch.randn(bs,1,8,8)).bernoulli_(p=0.5), scale_factor=64, mode='nearest').to(device)
                cqmix = res*mask + gt*(1-mask)    
                img_cat = torch.cat((res, gt, cqmix), dim=0)
                label_cat = torch.cat((label, label, label), dim=0)
                pred = clsnet(img_cat, normalize=True)
                loss_clsnet = F.cross_entropy(pred, label_cat, reduction="mean", label_smoothing=0.0)
            opt_clsnet.zero_grad()
            accelerator.backward(loss_clsnet)
            opt_clsnet.step(), sch_clsnet.step()
            
            accelerator.wait_for_everyone()

            global_step += 1
            loss_records["swinir_pix"].append(loss_swinir_pix.item())
            loss_records["swinir_tdp"].append(loss_swinir_tdp.item())
            loss_records["clsnet"].append(loss_clsnet.item())
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
                    writer.add_scalar("train/learning_rate_clsnet", opt_clsnet.param_groups[0]['lr'], global_step)

            # Save checkpoint
            if global_step % cfg.train.ckpt_every == 0 or (args.debug):
                if accelerator.is_local_main_process:
                    torch.save(pure_swinir.state_dict(), f"{ckpt_dir}/swinir_{global_step:07d}.pt")
                    torch.save(pure_clsnet.state_dict(), f"{ckpt_dir}/clsnet_{global_step:07d}.pt")

            # Save images
            if global_step % cfg.train.image_every == 0 or global_step == 1 or (args.debug):
                swinir.eval()
                with torch.no_grad():
                    N = 12
                    log_gt, log_lq = gt[:N], lq[:N]
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
                swinir.eval(), clsnet.eval()
                val_psnr, val_acc1 = [], []
                val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch",
                                total=len(val_loader), leave=False, desc="Validation")
                
                for val_gt, val_lq, val_label, _ in val_loader:
                    val_gt = rearrange(val_gt, "b h w c -> b c h w").contiguous().float().to(device)
                    val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float().to(device)
    
                    with torch.no_grad():
                        val_res = swinir(val_lq)
                        val_pred = clsnet(val_res, normalize=True)
                        val_res, val_gt, val_pred, val_label = accelerator.gather_for_metrics((val_res, val_gt, val_pred, val_label))
                        
                        if accelerator.is_local_main_process:
                            val_psnr += [calculate_psnr_pt(val_res, val_gt, crop_border=0).mean().item()] * val_gt.size(0)
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
                swinir.train(), clsnet.train()
            
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break
        
        pbar.close()
        epoch += 1

    # Save the last model weight
    if accelerator.is_local_main_process:
        torch.save(pure_swinir.state_dict(), f"{ckpt_dir}/swinir_last.pt")
        torch.save(pure_clsnet.state_dict(), f"{ckpt_dir}/clsnet_last.pt")
        Logging("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(args)
