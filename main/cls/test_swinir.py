import os
import sys
import torch
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from tqdm import tqdm
from model import SwinIR
from torch.nn import functional as F
from einops import rearrange
from argparse import ArgumentParser
from omegaconf import OmegaConf
from accelerate import Accelerator
from utils.common import (
    instantiate_from_config, load_network,
    calculate_psnr_pt, copy_opt_file, set_logger
)
from utils.classification import calculate_accuracy
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from torchvision.utils import save_image


def main(args) -> None:
    # Setup accelerator
    cfg = OmegaConf.load(args.config)
    accelerator = Accelerator(split_batches=True, mixed_precision=cfg.test.precision)
    set_seed(231)
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
    if cfg.test.resume_swinir is None:
        cfg.test.resume_swinir = os.path.join(exp_dir, 'checkpoints', 'swinir_last.pt')
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    load_path = cfg.test.resume_swinir if cfg.test.get('resume_swinir') else os.path.join(exp_dir, 'checkpoints', 'swinir_last.pt')
    swinir.load_state_dict(torch.load(load_path, map_location="cpu"), strict=True)
    Logging(f"Load SwinIR weight from checkpoint: {load_path}")
    
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
    teacher_clsnet.eval().to(device)
    clsnet.eval().to(device)
    swinir, teacher_clsnet, clsnet, val_loader = accelerator.prepare(swinir, teacher_clsnet, clsnet, val_loader)

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
        
        with torch.no_grad():
            # Restoration and classification
            val_res = swinir(val_lq)
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
            val_path = accelerator.gather_for_metrics(val_path)
            
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
    Logging("done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--save-img", action='store_true')
    parser.add_argument("--calc-fd", action='store_true')
    args = parser.parse_args()
    main(args)
