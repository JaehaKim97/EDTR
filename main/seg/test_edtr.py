import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
import utils.filter_warning

import math
import torch
from tqdm import tqdm
from model import SwinIR, ControlLDM, Diffusion
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.common import (
    instantiate_from_config, load_network,
    calculate_psnr_pt, wavelet_reconstruction
)
from utils.sampler import SpacedSampler
from utils.segmentation import (
    convert2color, calculate_mat, compute_iou, prepare_environment
)
from einops import rearrange
from argparse import ArgumentParser
from omegaconf import OmegaConf
from accelerate import Accelerator, DataLoaderConfiguration
from torchvision.utils import save_image


def main(args) -> None:
    # setup environment
    cfg = OmegaConf.load(args.config)
    accelerator = Accelerator(dataloader_config=DataLoaderConfiguration(split_batches=True),
                              mixed_precision=cfg.test.precision)
    device = accelerator.device
    dirs, Logging = prepare_environment(__name__, cfg, args, accelerator)
    exp_dir = dirs["exp"]
    if args.save_img:
        img_dir, mask_dir = dirs["img"], dirs["mask"]
    
    # create and load models
    if cfg.model.pre_restoration:
        swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
        if cfg.test.resume_swinir is None:
            cfg.test.resume_swinir = os.path.join(exp_dir, 'checkpoints', 'swinir_last.pt')
        swinir.load_state_dict(torch.load(cfg.test.resume_swinir, map_location="cpu"), strict=True)
        Logging(f"Load SwinIR weight from checkpoint: {cfg.test.resume_swinir}")
    else:
        Logging("Not using pre-restoration")
    
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.test.sd_path, map_location="cpu")["state_dict"]
    cldm.load_pretrained_sd(sd)
    if cfg.test.resume_cldm is None:
        cfg.test.resume_cldm = os.path.join(exp_dir, 'checkpoints', 'cldm_last.pt')
    cldm.load_controlnet_from_ckpt(torch.load(cfg.test.resume_cldm, map_location="cpu"))
    Logging(f"Load ControlNet weight from checkpoint: {cfg.test.resume_cldm}")
    if cfg.test.resume_decoder is None:
        cfg.test.resume_decoder = os.path.join(exp_dir, 'checkpoints', 'decoder_last.pt')
    cldm.vae.decoder.load_state_dict(torch.load(cfg.test.resume_decoder, map_location="cpu"))
    Logging(f"Load Decoder weight from checkpoint: {cfg.test.resume_decoder}")
    
    teacher_segnet = instantiate_from_config(cfg.model.segnet)  # segnet trained on HQ images
    if args.calc_fd:
        load_path = cfg.test.resume_teacher_segnet
        teacher_segnet = load_network(teacher_segnet, load_path, strict=True)
        Logging(f"Load Teacher SegmentationNetwork weight from checkpoint: {load_path}")
    
    if cfg.test.resume_segnet is None:
        cfg.test.resume_segnet = os.path.join(exp_dir, 'checkpoints', 'segnet_last.pt')
    segnet = instantiate_from_config(cfg.model.segnet)
    segnet = load_network(segnet, cfg.test.resume_segnet, strict=True)
    Logging(f"Load SegmentationNetwork weight from checkpoint: {cfg.test.resume_segnet}")
    
    # setup data
    val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False, drop_last=False
    )
    Logging(f"Validation dataset contains {len(val_dataset):,} images from {val_dataset.root}")
    
    # diffusion functions
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    sampler = SpacedSampler(diffusion.betas)
    val_ts, val_N = cfg.test.start_timestep, cfg.test.num_timesteps
    val_used_timesteps = [math.floor(val_ts/val_N*i) for i in range(1, val_N+1)]
    Logging(f"Used val timesteps are specified as {val_used_timesteps}, total number of {val_N}")

    # prepare models, testing logs
    swinir.eval().to(device)
    cldm.eval().to(device)
    teacher_segnet.eval().to(device)
    segnet.eval().to(device)
    diffusion.to(device)    
    swinir, cldm, teacher_segnet, segnet, val_loader = accelerator.prepare(swinir, cldm, teacher_segnet, segnet, val_loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    val_psnr, val_fd, n_classes = [], [], 21
    confmat = torch.zeros((n_classes, n_classes), device=device)
    
    # Testing:
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
            # padding
            h, w = val_lq.shape[2:]
            ph, pw = math.ceil(h/64)*64 - h, math.ceil(w/64)*64 - w
            val_lq_padded = F.pad(val_lq, pad=(0, pw, 0, ph), mode='replicate')
            
            # pre-restoration
            val_pre_res = val_lq
            if cfg.model.pre_restoration: val_pre_res = swinir(val_lq_padded)
            
            # prepare condition
            val_z_pre_res = pure_cldm.vae_encode(val_pre_res * 2 - 1, sample=False)
            val_cond = dict(c_txt=pure_cldm.clip.encode(val_prompt), c_img=val_z_pre_res)
            
            # partial diffusion
            val_noise = torch.randn_like(val_z_pre_res)
            val_t = torch.tensor([val_ts] * val_bs, dtype=torch.int64).to(device)
            val_z_noisy = diffusion.q_sample(x_start=val_z_pre_res, t=val_t, noise=val_noise)
            
            # short-step denoising
            val_z = sampler.manual_sample_with_timesteps(
                model=cldm, device=device, x_T=val_z_noisy, steps=len(val_used_timesteps),
                used_timesteps=val_used_timesteps, batch_size=val_bs, cond=val_cond, uncond=None,
                cfg_scale=1.0, progress=accelerator.is_local_main_process, progress_leave=False
            )
            val_res = wavelet_reconstruction((pure_cldm.vae_decode(val_z) + 1) / 2, val_pre_res)[:,:,:h,:w]
            
            # Segmentation
            val_pred = segnet(val_res)

            # calculate feature-distance
            if args.calc_fd:
                _, feat_gt = teacher_segnet(val_gt, return_feat=True)
                _, feat_res = teacher_segnet(val_res, return_feat=True)

            # save images
            if args.save_img and accelerator.is_local_main_process:
                for idx, basename in enumerate(val_path):
                    basename = os.path.basename(basename)
                    img_name = os.path.splitext(os.path.join(img_dir, basename))[0] + ".png"
                    save_image(val_res[idx], img_name)
                    
                    pred_mask = convert2color(val_pred["out"][idx:idx+1].argmax(1))
                    mask_name = os.path.splitext(os.path.join(mask_dir, basename))[0] + ".png"
                    save_image(pred_mask, mask_name)
            
            # calculate metrics
            val_psnr_batch = calculate_psnr_pt(val_res, val_gt, crop_border=0).unsqueeze(0)
            val_mat = calculate_mat(val_mask.flatten(), val_pred['out'].argmax(1).flatten(), n=n_classes).unsqueeze(0)
            val_psnr_batch, val_mat = accelerator.gather_for_metrics((val_psnr_batch, val_mat))
            if accelerator.is_local_main_process:
                for v in val_psnr_batch: val_psnr += [v.item()]
                confmat += val_mat.sum(0)
                if args.calc_fd:
                    val_fd += [F.l1_loss(input=feat_res["C5"], target=feat_gt["C5"], reduction="mean").item()]
            
            accelerator.wait_for_everyone()
        val_pbar.update(1)
    val_pbar.close()
    
    if accelerator.is_local_main_process:
        avg_val_psnr = torch.tensor(val_psnr).mean().item()
        avg_val_fd = torch.tensor(val_fd).mean().item()
        miou = compute_iou(confmat).mean().item() * 100
        for tag, val in [
            ("val/psnr", avg_val_psnr),
            ("val/fd", avg_val_fd),
            ("val/miou", miou),
        ]:
            if not ("val/fd" in tag) or args.calc_fd:
                Logging(f"{tag}: {val:.4f}")

    accelerator.wait_for_everyone()
    Logging("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--save-img", action='store_true')
    parser.add_argument("--calc-fd", action='store_true')
    args = parser.parse_args()
    main(args)
