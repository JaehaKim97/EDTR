import os
import sys
import torch
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from tqdm import tqdm
from utils.common import (
    instantiate_from_config, load_network,
    copy_opt_file, set_logger, calculate_psnr_pt
)
from utils.segmentation import convert2color, calculate_mat, compute_iou
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

    # Set up the experiment folder and logger
    exp_dir = cfg.test.exp_dir
    if accelerator.is_local_main_process:
        os.makedirs(exp_dir, exist_ok=True)
        logger = set_logger(__name__, exp_dir, logger_name=f"logger_test_s{args.seed}.log")
        copy_opt_file(args.config, exp_dir)
        print(f"Experiment directory created at {exp_dir}")
        if args.save_img and accelerator.is_local_main_process:
            pred_mask_dir = os.path.join(exp_dir, f'results_s{args.seed}', 'pred_mask')
            os.makedirs(pred_mask_dir, exist_ok=True)
            gt_mask_dir = os.path.join(exp_dir, f'results_s{args.seed}', 'gt_mask')
            os.makedirs(gt_mask_dir, exist_ok=True)
        Logging(f"Random seed: {args.seed}")

    # Create models
    segnet = instantiate_from_config(cfg.model.segnet)
    load_path = cfg.test.resume_segnet if cfg.test.get('resume_segnet') else os.path.join(exp_dir, 'checkpoints', 'segnet_last.pt')
    segnet = load_network(segnet, load_path, strict=True)
    Logging(f"Load SegmentationNetwork weight from checkpoint: {load_path}")
    
    # Setup data
    val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False, drop_last=False
    )
    Logging(f"Validation dataset contains {len(val_dataset):,} images from {val_dataset.root}")

    # Prepare models for testing
    segnet.eval().to(device)
    segnet, val_loader = accelerator.prepare(segnet, val_loader)

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
        
        # Input image type
        val_inp = val_gt if cfg.dataset.get('use_gt') else val_lq
        
        with torch.no_grad():
            val_pred = segnet(val_inp, normalize=True)
            val_mat = calculate_mat(val_mask.flatten(), val_pred['out'].argmax(1).flatten(), n=n_classes).unsqueeze(0)
            
            # Save images
            if args.save_img and accelerator.is_local_main_process:
                for idx, basename in enumerate(val_path):
                    basename = os.path.basename(basename)
                    pred_mask = convert2color(val_pred["out"][idx:idx+1].argmax(1))
                    pred_mask_name = os.path.splitext(os.path.join(pred_mask_dir, basename))[0] + ".png"
                    save_image(pred_mask, pred_mask_name)
                    
                    gt_mask = convert2color(val_mask[idx:idx+1])
                    gt_mask_name = os.path.splitext(os.path.join(gt_mask_dir, basename))[0] + ".png"
                    save_image(gt_mask, gt_mask_name)
                    
            # Calculate metrics
            val_psnr_batch = calculate_psnr_pt(val_inp, val_gt, crop_border=0).unsqueeze(0)
            val_mat = accelerator.gather_for_metrics(val_mat)
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
