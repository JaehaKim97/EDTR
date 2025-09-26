import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
import utils.filter_warning

import math
import torch
from tqdm import tqdm
from model import SwinIR
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.common import (
    instantiate_from_config, load_network,
    calculate_psnr_pt, wavelet_reconstruction
)
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

    # Create models
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    load_path = cfg.test.resume_swinir if cfg.test.get('resume_swinir') else os.path.join(exp_dir, 'checkpoints', 'swinir_last.pt')
    swinir.load_state_dict(torch.load(load_path, map_location="cpu"), strict=True)
    Logging(f"Load SwinIR weight from checkpoint: {load_path}")
    
    segnet = instantiate_from_config(cfg.model.segnet)
    load_path = cfg.test.resume_segnet if cfg.test.get('resume_segnet') else os.path.join(exp_dir, 'checkpoints', 'segnet_last.pt')
    segnet = load_network(segnet, load_path, strict=True)
    Logging(f"Load SegmentationNetwork weight from checkpoint: {load_path}")
    
    # setup data
    val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False, drop_last=False
    )
    Logging(f"Validation dataset contains {len(val_dataset):,} images from {val_dataset.root}")

    # prepare models, testing logs
    swinir.eval().to(device)
    segnet.eval().to(device)
    swinir, segnet, val_loader = accelerator.prepare(swinir, segnet, val_loader)
    val_psnr, n_classes = [], 21
    confmat = torch.zeros((n_classes, n_classes), device=device)

    # Testing:
    Logging(f"Testing start...")
    val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch",
                    total=len(val_loader), leave=False, desc="Validation")
    for val_gt, val_lq, val_mask, val_path in val_loader:
        val_gt = rearrange(val_gt, "b h w c -> b c h w").contiguous().float().to(device)
        val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float().to(device)
        val_mask = val_mask.long()
        assert(len(val_gt) == 1)
        
        with torch.no_grad():
            # padding
            h, w = val_lq.shape[2:]
            ph, pw = math.ceil(h/64)*64 - h, math.ceil(w/64)*64 - w
            val_lq_padded = F.pad(val_lq, pad=(0, pw, 0, ph), mode='replicate')
            
            # restoration and detection
            val_res = swinir(val_lq_padded)[:,:,:h,:w]
            val_pred = segnet(val_res, normalize=True)
            
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
                
            accelerator.wait_for_everyone()
        val_pbar.update(1)
    val_pbar.close()
 
    if accelerator.is_local_main_process:
        avg_val_psnr = torch.tensor(val_psnr).mean().item()
        miou = compute_iou(confmat).mean().item() * 100
        for tag, val in [
            ("val/psnr", avg_val_psnr),
            ("val/miou", miou),
        ]:
            Logging(f"{tag}: {val:.4f}")
          
    accelerator.wait_for_everyone()
    Logging("done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--save-img", action='store_true')
    args = parser.parse_args()
    main(args)
