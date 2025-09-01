import os
import sys
import math
import torch
import safetensors
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(parent_dir)

from tqdm import tqdm
from model import SwinIR, ControlLDM, Diffusion
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.common import (
    instantiate_from_config, wavelet_reconstruction,
    load_network, set_logger, copy_opt_file
)
from utils.detection import draw_box, collate_fn, list_to_batch, batch_to_list
from utils.sampler import SpacedSampler
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
        img_dir = os.path.join(exp_dir, f'results_s{args.seed}', 'img')
        os.makedirs(img_dir, exist_ok=True)
        box_dir = os.path.join(exp_dir, f'results_s{args.seed}', 'box')
        os.makedirs(box_dir, exist_ok=True)
        Logging(f"Random seed: {args.seed}")

    # Create models
    if cfg.model.pre_restoration:
        Logging("Using pre-restoration: SwinIR")
        swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
        swinir.load_state_dict(torch.load(cfg.test.resume_swinir, map_location="cpu"), strict=True)
        Logging(f"Load SwinIR weight from checkpoint: {cfg.test.resume_swinir}")
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
    
    if cfg.test.resume_detnet is None:
        cfg.test.resume_detnet = os.path.join(exp_dir, 'checkpoints', 'detnet_last.pt')
    detnet = instantiate_from_config(cfg.model.detnet)
    detnet = load_network(detnet, cfg.test.resume_detnet, strict=True)
    Logging(f"Load DetectionNetwork weight from checkpoint: {cfg.test.resume_detnet}")
    
    # Setup data
    val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers, shuffle=False, pin_memory=True, collate_fn=collate_fn
    )
    Logging(f"Validation dataset contains {len(val_dataset):,} images from {val_dataset.root}")
    
    # Diffusion functions
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    sampler = SpacedSampler(diffusion.betas)
    val_ts, val_N = cfg.test.start_timestep, cfg.test.num_timesteps
    val_used_timesteps = [math.floor(val_ts/val_N*i) for i in range(1, val_N+1)]
    Logging(f"Used val timesteps are specified as {val_used_timesteps}, total number of {val_N}")

    # Prepare models for testing
    swinir.eval().to(device)
    cldm.eval().to(device)
    detnet.eval().to(device)
    diffusion.to(device)    
    swinir, cldm, detnet, val_loader = accelerator.prepare(swinir, cldm, detnet, val_loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    
    # Testing
    if accelerator.mixed_precision == 'fp16':
        Logging("Mixed precision is applied")
    Logging(f"Testing start...")
    val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch",
                    total=len(val_loader), leave=False, desc="Validation")
    for img_list, img_path_list in val_loader:
        img_list = list(rearrange(img, 'h w c -> c h w').contiguous().float().to(device) for img in img_list)
        bs = len(img_list)
        prompt = [cfg.test.default_prompt] * bs
        
        with torch.no_grad():    
            # Pre-restoration
            img_batch = list_to_batch(img_list, img_size=512, device=device)
            pre_res_batch = swinir(img_batch)
            
            # Prepare condition
            z_pre_res = pure_cldm.vae_encode(pre_res_batch * 2 - 1, sample=False)
            cond = dict(c_txt=pure_cldm.clip.encode(prompt), c_img=z_pre_res)
            
            # Partial diffusion
            noise = torch.randn_like(z_pre_res)
            t = torch.tensor([val_ts] * bs, dtype=torch.int64).to(device)
            z_partial = diffusion.q_sample(x_start=z_pre_res, t=t, noise=noise)
            
            # Short-step denoising
            z = sampler.manual_sample_with_timesteps(
                model=cldm, device=device, x_T=z_partial, steps=len(val_used_timesteps), used_timesteps=val_used_timesteps,
                batch_size=bs, x_size=z_pre_res.shape[1:], cond=cond, uncond=None, cfg_scale=1.0, 
                progress=accelerator.is_local_main_process, progress_leave=False
            )
            res_batch = wavelet_reconstruction((pure_cldm.vae_decode(z) + 1) / 2, pre_res_batch)
            res_list = batch_to_list(res_batch, img_list)
            
            # Detection
            pred_list, _ = detnet(res_list)
            pred_list = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in pred_list]
            
            # Save images
            basename = os.path.basename(img_path_list[0])
            img_name = os.path.splitext(os.path.join(img_dir, basename))[0] + ".png"
            save_image(res_list[0], img_name)
            
            pred_box = draw_box(res_list[0], pred_list[0], score_threshold=0.8, split_acc=False)
            box_name = os.path.splitext(os.path.join(box_dir, basename))[0] + ".png"
            save_image(pred_box, box_name)
                
        accelerator.wait_for_everyone()
        val_pbar.update(1)
    val_pbar.close()
    
    accelerator.wait_for_everyone()
    Logging("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--seed", type=int, default=231)
    args = parser.parse_args()
    main(args)
