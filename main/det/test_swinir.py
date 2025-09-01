import os
import sys
import torch
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from tqdm import tqdm
from model import SwinIR
from utils.common import (
    instantiate_from_config, load_network, copy_opt_file,
    set_logger, calculate_psnr_pt
)
from utils.detection import (
    CocoEvaluator, draw_box, collate_fn, get_coco_api_from_dataset, _get_iou_types,
    list_to_batch, batch_to_list
)
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

    # Set up the experiment folder and logger
    exp_dir = cfg.test.exp_dir
    if accelerator.is_local_main_process:
        os.makedirs(exp_dir, exist_ok=True)
        logger = set_logger(__name__, exp_dir, logger_name=f"logger_test_s{args.seed}.log")
        copy_opt_file(args.config, exp_dir)
        print(f"Experiment directory created at {exp_dir}")
        if args.save_img and accelerator.is_local_main_process:
            img_dir = os.path.join(exp_dir, f'results_s{args.seed}', 'img')
            os.makedirs(img_dir, exist_ok=True)
            box_dir = os.path.join(exp_dir, f'results_s{args.seed}', 'box')
            os.makedirs(box_dir, exist_ok=True)
        Logging(f"Random seed: {args.seed}")

    # Create models
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    load_path = cfg.test.resume_swinir if cfg.test.get('resume_swinir') else os.path.join(exp_dir, 'checkpoints', 'swinir_last.pt')
    swinir.load_state_dict(torch.load(load_path, map_location="cpu"), strict=True)
    Logging(f"Load SwinIR weight from checkpoint: {load_path}")
    
    teacher_detnet = instantiate_from_config(cfg.model.detnet)  # detnet trained on HQ images
    if args.calc_fd:
        load_path = cfg.test.resume_teacher_detnet
        teacher_detnet = load_network(teacher_detnet, load_path, strict=True)
        Logging(f"Load Teacher DetectionNetwork weight from checkpoint: {load_path}")
    
    detnet = instantiate_from_config(cfg.model.detnet)
    load_path = cfg.test.resume_detnet if cfg.test.get('resume_detnet') else os.path.join(exp_dir, 'checkpoints', 'detnet_last.pt')
    detnet = load_network(detnet, load_path, strict=True)
    Logging(f"Load DetectionNetwork weight from checkpoint: {load_path}")
    
    # Setup data
    if cfg.dataset.get('use_gt'): Logging(f"Using ground-truth image!")
    val_dataset = instantiate_from_config(cfg.dataset.val)        
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers, shuffle=False, pin_memory=True, collate_fn=collate_fn
    )
    Logging(f"Validation dataset contains {len(val_dataset):,} images from {val_dataset.root}")

    # Prepare models for testing
    swinir.eval().to(device)
    teacher_detnet.eval().to(device)
    detnet.eval().to(device)
    swinir, teacher_detnet, detnet, val_loader = accelerator.prepare(swinir, teacher_detnet, detnet, val_loader)
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
        
        with torch.no_grad():
            # Restoration
            val_lq_batch = list_to_batch(val_lq_list, img_size=512, device=device)
            val_res_batch = swinir(val_lq_batch)
            val_res_list = batch_to_list(val_res_batch, val_gt_list)
            
            # Detection
            val_pred_list, _ = detnet(val_res_list)
            val_pred_list = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in val_pred_list]
            res = {annot["image_id"]: pred for annot, pred in zip(val_annot_list, val_pred_list)}
            
            # Calculate feature-distance
            if args.calc_fd:
                _, _, feat_gt = teacher_detnet(val_gt_list, return_feat=True)
                _, _, feat_res = teacher_detnet(val_res_list, return_feat=True)
            
            # Save images
            if args.save_img and accelerator.is_local_main_process:
                for idx, basename in enumerate(val_path_list):
                    basename = os.path.basename(val_path_list[idx])
                    img_name = os.path.splitext(os.path.join(img_dir, basename))[0] + ".png"
                    save_image(val_res_list[idx], img_name)
                    
                    val_pred_box = draw_box(val_res_list[idx], val_pred_list[idx],
                                            score_threshold=0.6, fontsize=0.7, split_acc=True)
                    box_name = os.path.splitext(os.path.join(box_dir, basename))[0] + ".png"
                    save_image(val_pred_box, box_name)
            
            # Calculate metrics
            val_gt, val_res = val_gt_list[0].unsqueeze(0), val_res_list[0].unsqueeze(0)
            _val_psnr = calculate_psnr_pt(val_res, val_gt, crop_border=0).unsqueeze(0)
            _val_psnr = accelerator.gather_for_metrics((_val_psnr))
            if accelerator.is_local_main_process:
                for v in _val_psnr: val_psnr += [v.item()]
                if args.calc_fd:
                    val_fd += [(F.l1_loss(input=feat_res['features']['0'], target=feat_gt['features']['0'], reduction="mean") * 0.5 + \
                        F.l1_loss(input=feat_res['features']['1'], target=feat_gt['features']['1'], reduction="mean") * 0.5).item()]
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
