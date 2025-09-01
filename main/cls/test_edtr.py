import os
import sys
import math
import torch
import safetensors
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from tqdm import tqdm
from utils.common import (
    instantiate_from_config, wavelet_reconstruction,
    load_network, set_logger, copy_opt_file, calculate_psnr_pt
)
from utils.classification import calculate_accuracy
from utils.sampler import SpacedSampler
from model import SwinIR, ControlLDM, Diffusion
from torch.nn import functional as F
from torch.utils.data import DataLoader
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
        Logging(f"Random seed: {args.seed}")
    
    # Create models    
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    swinir.load_state_dict(torch.load(cfg.test.resume_swinir, map_location="cpu"), strict=True)
    Logging(f"Load SwinIR weight from checkpoint: {cfg.test.resume_swinir}")
    if cfg.model.pre_restoration:
        Logging("Using pre-restoration: SwinIR")
    else:
        Logging("Not using pre-restoration")
    
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
    
    if cfg.test.resume_decoder is None:
        cfg.test.resume_decoder = os.path.join(exp_dir, 'checkpoints', 'decoder_last.pt')
    cldm.vae.decoder.load_state_dict(torch.load(cfg.test.resume_decoder, map_location="cpu"))
    Logging(f"Load Decoder weight from checkpoint: {cfg.test.resume_decoder}")
    
    teacher_clsnet = instantiate_from_config(cfg.model.clsnet)  # clsnet trained on HQ images
    if args.calc_fd:
        load_path = cfg.test.resume_teacher_clsnet
        teacher_clsnet = load_network(teacher_clsnet, load_path, strict=True)
        Logging(f"Load Teacher ClassificationNetwork weight from checkpoint: {load_path}")
    
    if cfg.test.resume_clsnet is None:
        cfg.test.resume_clsnet = os.path.join(exp_dir, 'checkpoints', 'clsnet_last.pt')
    clsnet = instantiate_from_config(cfg.model.clsnet)
    clsnet = load_network(clsnet, cfg.test.resume_clsnet, strict=True)
    Logging(f"Load ClassificationNetwork weight from checkpoint: {cfg.test.resume_clsnet}")
    
    # Diffusion functions
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    sampler = SpacedSampler(diffusion.betas)
    val_ts, val_N = cfg.test.start_timestep, cfg.test.num_timesteps
    val_used_timesteps = [math.floor(val_ts/val_N*i) for i in range(1, val_N+1)]
    Logging(f"Used val timesteps are specified as {val_used_timesteps}, total number of {val_N}")
    
    # Setup data
    val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False, drop_last=False
    )
    Logging(f"Validation dataset contains {len(val_dataset):,} images from {val_dataset.root}")

    # Prepare models for testing
    swinir.eval().to(device)
    cldm.eval().to(device)
    teacher_clsnet.eval().to(device)
    clsnet.eval().to(device)
    diffusion.to(device)    
    swinir, cldm, teacher_clsnet, clsnet, val_loader = accelerator.prepare(swinir, cldm, teacher_clsnet, clsnet, val_loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
        
    # Evaluation
    if accelerator.mixed_precision == 'fp16':
        Logging("Mixed precision is applied")
    val_psnr, val_acc1, val_fd = [], [], []
    Logging(f"Testing start...")
    val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch",
                    total=len(val_loader), leave=False, desc="Validation")
    for val_gt, val_lq, val_label, val_path in val_loader:
        val_gt = rearrange(val_gt, "b h w c -> b c h w").contiguous().float().to(device)
        val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float().to(device)
        val_bs = val_gt.size(0)
        val_prompt = [cfg.test.default_prompt] * val_bs
        
        with torch.no_grad():
            # Pre-restoration
            val_pre_res = val_lq
            if cfg.model.pre_restoration: val_pre_res = swinir(val_lq)
            
            # Prepare condition
            val_z_pre_res = pure_cldm.vae_encode(val_pre_res * 2 - 1, sample=False)
            val_cond = dict(c_txt=pure_cldm.clip.encode(val_prompt), c_img=val_z_pre_res)
            
            # Partial diffusion
            val_noise = torch.randn_like(val_z_pre_res)
            val_t = torch.tensor([val_ts] * val_bs, dtype=torch.int64).to(device)
            val_z_partial = diffusion.q_sample(x_start=val_z_pre_res, t=val_t, noise=val_noise)

            # Short-step denoising
            val_z = sampler.manual_sample_with_timesteps(
                model=cldm, device=device, x_T=val_z_partial, steps=len(val_used_timesteps), used_timesteps=val_used_timesteps,
                batch_size=val_bs, x_size=val_z_pre_res.shape[1:], cond=val_cond, uncond=None, cfg_scale=1.0, 
                progress=accelerator.is_local_main_process, progress_leave=False
            )
            val_res = wavelet_reconstruction((pure_cldm.vae_decode(val_z) + 1) / 2, val_pre_res)
            
            # Classification
            val_pred = clsnet(val_res, normalize=True)
            
            # Calculate feature-distance
            if args.calc_fd:
                _, feat_gt = teacher_clsnet(val_gt, return_feat=True)
                _, feat_res = teacher_clsnet(val_res, return_feat=True)
                val_gt, val_res, val_pred, val_label, feat_gt, feat_res = \
                    accelerator.gather_for_metrics((val_gt, val_res, val_pred, val_label, feat_gt, feat_res))
            else:
                val_gt, val_res, val_pred, val_label = \
                    accelerator.gather_for_metrics((val_gt, val_res, val_pred, val_label))
            
            # Save images
            if args.save_img and accelerator.is_local_main_process:
                for idx, basename in enumerate(val_path):
                    basename = "{:03d}.".format(val_label[idx]+1) + os.path.basename(basename)
                    cls_cmp = "_<GT{:03d}-Pred{:03d}>".format(val_label[idx]+1, val_pred.argmax(1)[idx]+1)
                    img_name = os.path.splitext(os.path.join(img_dir, basename))[0] + cls_cmp + ".png"
                    save_image(val_res[idx], img_name)
            
            # Calculate metrics
            if accelerator.is_local_main_process:
                val_psnr += calculate_psnr_pt(val_res, val_gt, crop_border=0).tolist()
                val_acc1 += [calculate_accuracy(val_pred, val_label, topk=(1, 5))[0]] * val_gt.size(0)
                if args.calc_fd:
                    val_fd += F.l1_loss(input=feat_res, target=feat_gt, reduction='none').mean(dim=(1,2,3)).tolist()
            
            accelerator.wait_for_everyone()
        val_pbar.update(1)
    val_pbar.close()
    
    if accelerator.is_local_main_process:
        avg_val_psnr = torch.tensor(val_psnr).mean().item()
        avg_val_acc1 = torch.tensor(val_acc1).mean().item()
        avg_val_fd = torch.tensor(val_fd).mean().item()
        for tag, val in [
            ("val/psnr", avg_val_psnr),
            ("val/acc1", avg_val_acc1),
            ("val/fd", avg_val_fd),
        ]:
            if not ("val/fd" in tag) or args.calc_fd:
                Logging(f"{tag}: {val:.4f}")

    accelerator.wait_for_everyone()
    Logging("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save-img", action='store_true')
    parser.add_argument("--calc-fd", action='store_true')
    args = parser.parse_args()
    main(args)
