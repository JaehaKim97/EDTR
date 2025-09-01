import os
import sys
import math
import torch
import safetensors
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from tqdm import tqdm
from model import SwinIR, ControlLDM, Diffusion
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.common import (
    instantiate_from_config, wavelet_reconstruction, load_network,
    set_logger, calculate_psnr_pt, copy_opt_file
)
from utils.sampler import SpacedSampler
from utils.segmentation import convert2color, calculate_mat, compute_iou
from einops import rearrange
from argparse import ArgumentParser
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision.utils import save_image


def main(args) -> None:
    # Setup accelerator
    cfg = OmegaConf.load(args.config)
    accelerator = Accelerator(split_batches=True, mixed_precision=cfg.test.precision)
    set_seed(args.seed)
    device = accelerator.device
    
    def Logging(text, print=True):
        if accelerator.is_local_main_process:
            if print:
                logger.info(text)
            else:
                logger.debug(text)

    # Setup an experiment folder
    exp_dir = cfg.test.exp_dir
    if accelerator.is_local_main_process:
        os.makedirs(exp_dir, exist_ok=True)
        logger = set_logger(__name__, exp_dir, logger_name=f"logger_test_s{args.seed}.log")
        copy_opt_file(args.config, exp_dir)
        Logging(f"Experiment directory created at {exp_dir}")
        if args.save_img and accelerator.is_local_main_process:
            img_dir = os.path.join(exp_dir, f'results_s{args.seed}', 'img')
            os.makedirs(img_dir, exist_ok=True)
            mask_dir = os.path.join(exp_dir, f'results_s{args.seed}', 'mask')
            os.makedirs(mask_dir, exist_ok=True)
        Logging(f"Random seed: {args.seed}")

    # Create models
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    swinir.load_state_dict(torch.load(cfg.test.resume_swinir, map_location="cpu"), strict=True)
    Logging(f"Load SwinIR weight from checkpoint: {cfg.test.resume_swinir}")
    
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    if 'turbo' in cfg.test.sd_path:
        sd = safetensors.torch.load_file(cfg.test.sd_path, device="cpu")
        unused = cldm.load_pretrained_sd(sd, is_turbo=True)
        Logging(f"Stable Diffusion **TURBO** is using")    
    else:
        sd = torch.load(cfg.test.sd_path, map_location="cpu")["state_dict"]
        unused = cldm.load_pretrained_sd(sd)
    Logging(f"Load pretrained SD weight from {cfg.test.sd_path}")
        
    if cfg.test.resume_cldm is None:
        cfg.test.resume_cldm = os.path.join(exp_dir, 'checkpoints', 'cldm_last.pt')
    cldm.load_controlnet_from_ckpt(torch.load(cfg.test.resume_cldm, map_location="cpu"))
    Logging(f"Load ControlNet weight from checkpoint: {cfg.test.resume_cldm}")
    
    if cfg.test.resume_segnet is None:
        cfg.test.resume_segnet = os.path.join(exp_dir, 'checkpoints', 'segnet_last.pt')
    segnet = instantiate_from_config(cfg.model.segnet)
    segnet = load_network(segnet, cfg.test.resume_segnet, strict=True)
    Logging(f"Load SegmentationNetwork weight from checkpoint: {cfg.test.resume_segnet}")
    
    # Setup data
    val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False, drop_last=False
    )
    Logging(f"Validation dataset contains {len(val_dataset):,} images from {val_dataset.root}")
    
    # Diffusion functions
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    sampler = SpacedSampler(diffusion.betas)

    # Prepare models for testing
    swinir.eval().to(device)
    cldm.eval().to(device)
    segnet.eval().to(device)
    diffusion.to(device)    
    swinir, cldm, segnet, val_loader = accelerator.prepare(swinir, cldm, segnet, val_loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    
    # Evaluation
    if accelerator.mixed_precision == 'fp16':
        Logging("Mixed precision is applied")
    val_psnr, n_classes = [], 21
    confmat = torch.zeros((n_classes, n_classes), device=device)
    Logging(f"Testing start...")
    val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch",
                    total=len(val_loader), leave=False, desc="Validation")
    for val_gt, val_lq, val_mask, val_path in val_loader:
        val_gt = rearrange(val_gt, "b h w c -> b c h w").contiguous().float().to(device)
        val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float().to(device)
        val_mask = val_mask.long()
        val_bs = val_gt.size(0)
        val_prompt = [cfg.test.default_prompt] * val_bs
        
        assert (val_bs == 1)
        
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
            
            # Save images
            if args.save_img and accelerator.is_local_main_process:
                for idx, basename in enumerate(val_path):
                    basename = os.path.basename(basename)
                    img_name = os.path.splitext(os.path.join(img_dir, basename))[0] + ".png"
                    save_image(val_res[idx], img_name)
                    
                    pred_mask = convert2color(val_pred["out"][idx:idx+1].argmax(1))
                    mask_name = os.path.splitext(os.path.join(mask_dir, basename))[0] + ".png"
                    save_image(pred_mask, mask_name)
            
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
            ("val/psnr", avg_val_psnr),
            ("val/miou", miou),
        ]:
            Logging(f"{tag}: {val:.4f}")
    
    accelerator.wait_for_everyone()
    Logging("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--save-img", action='store_true')
    args = parser.parse_args()
    main(args)
