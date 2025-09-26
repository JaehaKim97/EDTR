import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
import utils.filter_warning

import torch
from tqdm import tqdm
from utils.common import (
    instantiate_from_config, load_network,
    calculate_psnr_pt, wavelet_reconstruction
)
from utils.classification import calculate_accuracy, prepare_environment
from torch.nn import functional as F
from torch.utils.data import DataLoader
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
        img_dir = dirs["img"]
        
    # create and load models
    teacher_clsnet = instantiate_from_config(cfg.model.clsnet)  # clsnet trained on HQ images
    if args.calc_fd:
        load_path = cfg.test.resume_teacher_clsnet
        teacher_clsnet = load_network(teacher_clsnet, load_path, strict=True)
        Logging(f"Load Teacher ClassificationNetwork weight from checkpoint: {load_path}")
    
    clsnet = instantiate_from_config(cfg.model.clsnet)
    load_path = cfg.test.resume_clsnet if cfg.test.get('resume_clsnet') else os.path.join(exp_dir, 'checkpoints', 'clsnet_last.pt')
    if os.path.exists(load_path) or "ResNet" in load_path:
        clsnet = load_network(clsnet, load_path, strict=cfg.test.strict_load)
        Logging(f"Load ClassificationNetwork weight from checkpoint: {load_path}, strict: {cfg.test.strict_load}")
    else:
        Logging("Initialize ClassificationNetwork from scratch")
        
    # setup data
    val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False, drop_last=False
    )
    Logging(f"Validation dataset contains {len(val_dataset):,} images from {val_dataset.root}")

    # prepare models, testing logs
    clsnet.eval().to(device)
    teacher_clsnet.eval().to(device)
    teacher_clsnet, clsnet, val_loader = accelerator.prepare(teacher_clsnet, clsnet, val_loader)
    val_psnr, val_acc1, val_fd = [], [], []

    # Testing:
    Logging(f"Testing start...")
    val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch",
                    total=len(val_loader), leave=False, desc="Validation")
    for val_gt, val_lq, val_label, val_path in val_loader:
        val_gt = rearrange(val_gt, "b h w c -> b c h w").contiguous().float().to(device)
        val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float().to(device)
        
        # input image type
        val_inp = val_gt if cfg.dataset.get('use_gt') else val_lq
        
        with torch.no_grad():
            val_pred = clsnet(val_inp, normalize=True)
            
            # calculate feature-distance
            if args.calc_fd:  
                _, feat_gt = teacher_clsnet(val_gt, return_feat=True)
                _, feat_inp = teacher_clsnet(val_inp, return_feat=True)
                val_pred, val_label, feat_gt, feat_inp = \
                    accelerator.gather_for_metrics((val_pred, val_label, feat_gt, feat_inp))
            else:
                val_pred, val_label = accelerator.gather_for_metrics((val_pred, val_label))
            val_path = accelerator.gather_for_metrics(val_path)
            
            # save images
            if args.save_img and accelerator.is_local_main_process:
                for idx, basename in enumerate(val_path):
                    basename = "{:03d}.".format(val_label[idx]+1) + os.path.basename(basename)
                    cls_cmp = "_<GT{:03d}-Pred{:03d}>".format(val_label[idx]+1, val_pred.argmax(1)[idx]+1)
                    img_name = os.path.splitext(os.path.join(img_dir, basename))[0] + cls_cmp + ".png"
                    save_image(val_inp[idx], img_name)
            
            # calculate metrics
            if accelerator.is_local_main_process:
                val_psnr += calculate_psnr_pt(val_inp, val_gt, crop_border=0).tolist()
                val_acc1 += [calculate_accuracy(val_pred, val_label, topk=(1, 5))[0]] * val_pred.size(0)
                if args.calc_fd:
                    val_fd += F.l1_loss(input=feat_inp, target=feat_gt, reduction='none').mean(dim=(1,2,3)).tolist()
                    
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
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--save-img", action='store_true')
    parser.add_argument("--calc-fd", action='store_true')
    args = parser.parse_args()
    main(args)
