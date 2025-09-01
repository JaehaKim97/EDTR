import os
import sys
import torch
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from tqdm import tqdm
from einops import rearrange
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.common import (
    instantiate_from_config, load_network, copy_opt_file,
    set_logger, calculate_psnr_pt
)
from utils.detection import (
    CocoEvaluator, draw_box, collate_fn, get_coco_api_from_dataset, _get_iou_types,
)
from argparse import ArgumentParser
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
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
        img_dir = os.path.join(exp_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        logger = set_logger(__name__, exp_dir, logger_name=f"logger_test_s{args.seed}.log")
        copy_opt_file(args.config, exp_dir)
        Logging(f"Experiment directory created at {exp_dir}")
        if args.save_img and accelerator.is_local_main_process:
            pred_box_dir = os.path.join(exp_dir, f'results_s{args.seed}', 'pred_box')
            os.makedirs(pred_box_dir, exist_ok=True)
            gt_box_dir = os.path.join(exp_dir, f'results_s{args.seed}', 'gt_box')
            os.makedirs(gt_box_dir, exist_ok=True)
        Logging(f"Random seed: {args.seed}")

    # Create models
    teacher_detnet = instantiate_from_config(cfg.model.detnet)  # detnet trained on HQ images
    if args.calc_fd:
        load_path = cfg.test.resume_teacher_detnet
        teacher_detnet = load_network(teacher_detnet, load_path, strict=True)
        Logging(f"Load Teacher DetectionNetwork weight from checkpoint: {load_path}")
    
    if cfg.test.resume_detnet is None:
        cfg.test.resume_detnet = os.path.join(exp_dir, 'checkpoints', 'detnet_last.pt')
    detnet = instantiate_from_config(cfg.model.detnet)
    detnet = load_network(detnet, cfg.test.resume_detnet, strict=cfg.test.strict_load)
    Logging(f"Load DetectionNetwork weight from checkpoint: {cfg.test.resume_detnet}")
        
    # Setup data
    if cfg.dataset.get('use_gt'): Logging(f"Using ground-truth image!")
    val_dataset = instantiate_from_config(cfg.dataset.val)        
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers, shuffle=False, pin_memory=True, collate_fn=collate_fn
    )
    Logging(f"Validation dataset contains {len(val_dataset):,} images from {val_dataset.root}")

    # Prepare models for testing
    teacher_detnet.eval().to(device)
    detnet.eval().to(device)
    teacher_detnet, detnet, val_loader = accelerator.prepare(teacher_detnet, detnet, val_loader)
    pure_detnet = accelerator.unwrap_model(detnet)
    
    # Evaluation
    if accelerator.mixed_precision == 'fp16':
        Logging("Mixed precision is applied")
    val_psnr, val_fd = [], []
    Logging("Preparing COCO API for evaluation ..")
    coco = get_coco_api_from_dataset(val_dataset)
    iou_types = _get_iou_types(pure_detnet)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    Logging(f"Testing start...")
    val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch",
                    total=len(val_loader), leave=False, desc="Validation")
    for val_gt_list, val_lq_list, val_annot_list, val_path_list in val_loader:
        val_gt_list = list(rearrange(val_gt, 'h w c -> c h w').contiguous().float().to(device) for val_gt in val_gt_list)
        val_lq_list = list(rearrange(val_lq, 'h w c -> c h w').contiguous().float().to(device) for val_lq in val_lq_list)
        val_annot_list = [{k: v.long().to(device) if isinstance(v, torch.Tensor) else v for k, v in val_t.items()} for val_t in val_annot_list]
        assert (len(val_gt_list) == 1)
        
        # Input image type
        val_inp_list = val_gt_list if cfg.dataset.get('use_gt') else val_lq_list
        
        with torch.no_grad():
            val_pred_list, _ = detnet(val_inp_list)
            val_pred_list = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in val_pred_list]
            
            # Calculate feature-distance
            if args.calc_fd:
                _, _, feat_gt = teacher_detnet(val_gt_list, return_feat=True)
                _, _, feat_inp = teacher_detnet(val_inp_list, return_feat=True)
            
            # Save images
            if args.save_img and accelerator.is_local_main_process:
                for idx, basename in enumerate(val_path_list):
                    basename = os.path.basename(basename)
                    val_pred_box = draw_box(val_inp_list[idx], val_pred_list[idx],
                                            score_threshold=0.6, fontsize=0.7, split_acc=True)
                    pred_box_name = os.path.splitext(os.path.join(pred_box_dir, basename))[0] + ".png"
                    save_image(val_pred_box, pred_box_name)
                    
                    val_gt_box = draw_box(val_gt_list[idx], val_annot_list[idx])
                    gt_box_name = os.path.splitext(os.path.join(gt_box_dir, basename))[0] + ".png"
                    save_image(val_gt_box, gt_box_name)
            
            # Calculate metrics
            val_gt, val_inp = val_gt_list[0].unsqueeze(0), val_inp_list[0].unsqueeze(0)
            val_psnr_batch = calculate_psnr_pt(val_inp, val_gt, crop_border=0).unsqueeze(0)
            res = {annot["image_id"]: pred for annot, pred in zip(val_annot_list, val_pred_list)}
            if accelerator.is_local_main_process:
                for v in val_psnr_batch: val_psnr += [v.item()]
                if args.calc_fd:
                    val_fd += [(F.l1_loss(input=feat_inp['features']['0'], target=feat_gt['features']['0'], reduction="mean") * 0.5 + \
                        F.l1_loss(input=feat_inp['features']['1'], target=feat_gt['features']['1'], reduction="mean") * 0.5).item()]
            coco_evaluator.update(res)
            
        accelerator.wait_for_everyone()
        val_pbar.update(1)
    val_pbar.close()
    
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    
    if accelerator.is_local_main_process:
        avg_val_psnr = torch.tensor(val_psnr).mean().item()
        avg_val_fd = torch.tensor(val_fd).mean().item()
        det_results = coco_evaluator.summarize(Logging=Logging)
        for tag, val in [
            ("val/psnr", avg_val_psnr),
            ("val/fd", avg_val_fd),
            ("val/mAP@[0.5:0.95]", det_results["mAP@[0.5:0.95]"]),
            ("val/mAP@0.5", det_results["mAP@0.5"]),
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
