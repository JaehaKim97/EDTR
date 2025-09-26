import os
import math
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from glob import glob
from tqdm import tqdm
from utils.common import (
    instantiate_from_config, wavelet_reconstruction,
    pad_if_smaller, pad_to_multiples_of, make_tiled_fn
)
from utils.detection import draw_box, sliding_windows, move_boxes
from utils.sampler import SpacedSampler
from einops import rearrange
from argparse import ArgumentParser
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision.ops import batched_nms
from torchvision.utils import save_image


def main(args) -> None:
    # Setup accelerator
    cfg = OmegaConf.load("configs/det/demo.yaml")
    accelerator = Accelerator(mixed_precision=args.precision)
    set_seed(args.seed)
    device = accelerator.device
    is_coco = True if cfg.dataset.get('is_coco') else False
    
    # Setup an experiment folder
    img_dir = os.path.join(f'{args.output}/img')
    box_dir = os.path.join(f'{args.output}/box')
    os.makedirs(box_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    print(f"Random seed: {args.seed}")

    # Create and loading models
    swinir = instantiate_from_config(cfg.model.swinir)
    cldm = instantiate_from_config(cfg.model.cldm)
    detnet = instantiate_from_config(cfg.model.detnet)
    
    sd_weight = torch.load("weights/v2-1_512-ema-pruned.ckpt", map_location="cpu")["state_dict"]
    cldm.load_pretrained_sd(sd_weight)
    edtr_path = "weights/edtr-r50v2-coco-resrgan.pt"
    print(f"Used EDTR weight: {edtr_path}")
    try: edtr_weight = torch.load(edtr_path, map_location="cpu")
    except: print(f"Error: pre-trained EDTR weights not found at {edtr_path}")
    swinir.load_state_dict(edtr_weight["swinir"], strict=True)
    cldm.load_controlnet_from_ckpt(edtr_weight["cldm"])
    cldm.vae.decoder.load_state_dict(edtr_weight["decoder"])
    detnet.load_state_dict(edtr_weight["detnet"], strict=True)
    
    # Prepare images
    exts = ["png", "jpg", "jpeg", "JPG", "JPEG"]
    img_paths = sorted(sum([glob(os.path.join(args.input, f"*.{e}")) for e in exts], []))
    
    # diffusion functions
    diffusion = instantiate_from_config(cfg.model.diffusion)
    sampler = SpacedSampler(diffusion.betas)
    val_ts, val_N = cfg.test.start_timestep, cfg.test.num_timesteps
    val_used_timesteps = [math.floor(val_ts/val_N*i) for i in range(1, val_N+1)]

    # prepare models for testing
    swinir.eval().to(device)
    cldm.eval().to(device)
    detnet.eval().to(device)
    diffusion.to(device)    
    swinir, cldm, detnet = accelerator.prepare(swinir, cldm, detnet)
    pure_cldm = accelerator.unwrap_model(cldm)
    
    # testing
    if accelerator.mixed_precision == 'fp16':
        print("Mixed precision is applied")
    
    print(f"Testing start...")
    pbar = tqdm(img_paths)
    for img_path in pbar:
        pbar.set_description(f"Processing {img_path}")
        img = Image.open(img_path).convert("RGB")
        if args.scale == -1.0:
            scale = 512 / max(img.size[0], img.size[1])
            img = img.resize((int(round(x * scale)) for x in img.size), Image.BICUBIC)
        else:
            img = img.resize((int(x * args.scale) for x in img.size), Image.BICUBIC)
        img = torch.Tensor((np.array(img) / 255.0).astype(np.float32))
        img = rearrange(img, 'h w c -> c h w').contiguous().float().to(device).unsqueeze(0)
        h0, w0 = img.shape[2:]  # output size
        
        img = pad_if_smaller(img, size=512)
        img = pad_to_multiples_of(img, multiple=64)
        
        with torch.no_grad(), accelerator.autocast():
            # pre-restoration
            if args.pre_res_tiled:
                if max(img.shape[2:]) < 512:
                    print("[Pre-restoration] Input size is tiny and unnecessary to tile")
                else:
                    swinir = make_tiled_fn(swinir, size=512, stride=256)
            pre_res = swinir(img)
            
            # prepare condition
            z_pre_res = pure_cldm.vae_encode(pre_res * 2 - 1, sample=False,
                                             tiled=args.vae_encoder_tiled, tile_size=args.vae_encoder_tile_size)
            cond = dict(c_txt=pure_cldm.clip.encode(text=""), c_img=z_pre_res)
            
            # partial diffusion
            noise = torch.randn_like(z_pre_res)
            t = torch.tensor([val_ts], dtype=torch.int64).to(device)
            z_partial = diffusion.q_sample(x_start=z_pre_res, t=t, noise=noise)
            
            # short-step denoising
            h1, w1 = z_pre_res.shape[2:]
            cldm_tiled = args.cldm_tiled
            if cldm_tiled and (h1 <= args.cldm_tile_size // 8 or w1 <= args.cldm_tile_size // 8):
                print("[Diffusion]: the input size is tiny and unnecessary to tile.")
                cldm_tiled = False
            z = sampler.manual_sample_with_timesteps(
                model=cldm, device=device, x_T=z_partial, steps=len(val_used_timesteps),
                used_timesteps=val_used_timesteps, batch_size=1, cond=cond, uncond=None, cfg_scale=1.0,
                tiled=cldm_tiled, tile_size=args.cldm_tile_size//8, tile_stride=args.cldm_tile_stride//8,
                progress=accelerator.is_local_main_process, progress_leave=False
            )
            res = (pure_cldm.vae_decode(z, tiled=args.vae_decoder_tiled, tile_size=args.vae_decoder_tile_size) + 1) / 2
            res = wavelet_reconstruction(res, pre_res)[0]
            
            # Detection supports three modes: "tile", "resize", and "direct".
            # Default is "resize", but it may reduce performance on small objects.
            # In such cases, consider using "tile" or "direct".
            det_type = args.detection_type
            if det_type == "resize":
                scale = 512 / max(res.shape[2:])
                res_resize = F.interpolate(res.unsqueeze(0), scale_factor=scale, mode="bilinear", align_corners=False)[0]
                pred_list, _ = detnet([res_resize])
                pred_list = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in pred_list]
                pred_list[0]["boxes"] /= scale
            elif det_type == "tile":
                all_boxes, all_scores, all_labels = [], [], []
                h2, w2 = res.shape[1:]
                for (wx, hx, wy, hy) in sliding_windows(w2, h2, tile=args.detection_tile_size,
                                                        stride=args.detection_tile_stride):
                    pred_list, _ = detnet([res[:, hx:hy, wx:wy]])
                    out = pred_list[0]
                    keep = out["scores"] >= 0.6
                    if keep.any():
                        boxes, scores, labels = out["boxes"][keep], out["scores"][keep], out["labels"][keep]
                    boxes = move_boxes(boxes, dx=wx, dy=hx)
                    all_boxes.append(boxes.cpu()), all_scores.append(scores.cpu()), all_labels.append(labels.cpu())
                
                if len(all_boxes) == 0:
                    return dict(boxes=torch.zeros((0,4)), scores=torch.zeros((0,)),
                                labels=torch.zeros((0,), dtype=torch.long))
                    
                all_boxes = torch.cat(all_boxes, dim=0)
                all_scores = torch.cat(all_scores, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                keep = batched_nms(all_boxes, all_scores, all_labels, iou_threshold=args.detection_tile_nms_threshold)
                pred_list = [dict(boxes=all_boxes[keep], scores=all_scores[keep], labels=all_labels[keep])]
            else:
                pred_list, _ = detnet([res])
                pred_list = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in pred_list]
            
            # save images
            basename = os.path.basename(img_path)
            img_name = os.path.splitext(os.path.join(img_dir, basename))[0] + ".png"
            save_image(res[:,:h0, :w0], img_name)
            
            pred_box = draw_box(res, pred_list[0], score_threshold=args.threshold, split_acc=True, is_coco=is_coco)
            box_name = os.path.splitext(os.path.join(box_dir, basename))[0] + ".png"
            save_image(pred_box[:,:h0,:w0], box_name)
    
    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--seed", type=int, default=217)
    parser.add_argument("--precision", type=str, default=None, choices=[None, "fp16"])
    parser.add_argument("--scale", type=float, default=-1.0)
    parser.add_argument("--threshold", type=float, default=0.8)
    # tiling options
    parser.add_argument("--pre-res-tiled", action='store_true')
    parser.add_argument("--pre-res-tile-size", type=int, default=512)
    parser.add_argument("--pre-res-tile-stride", type=int, default=256)
    parser.add_argument("--vae-encoder-tiled", action='store_true')
    parser.add_argument("--vae-encoder-tile-size", type=int, default=256)
    parser.add_argument("--vae-decoder-tiled", action='store_true')
    parser.add_argument("--vae-decoder-tile-size", type=int, default=256)
    parser.add_argument("--cldm-tiled", action='store_true')
    parser.add_argument("--cldm-tile-size", type=int, default=512)
    parser.add_argument("--cldm-tile-stride", type=int, default=256)
    # detection options
    parser.add_argument("--detection-type", type=str, default="resize", choices=["resize", "tile", "direct"])
    parser.add_argument("--detection-tile-size", type=int, default=512)
    parser.add_argument("--detection-tile-stride", type=int, default=256)
    parser.add_argument("--detection-tile-nms-threshold", type=float, default=0.3)
    args = parser.parse_args()
    main(args)
