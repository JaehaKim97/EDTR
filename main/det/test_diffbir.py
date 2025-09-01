import os
import sys
import torch
import safetensors
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from tqdm import tqdm
from model import SwinIR, ControlLDM, Diffusion
from torch.utils.data import DataLoader
from utils.common import (
    instantiate_from_config, wavelet_reconstruction,
    load_network, set_logger, calculate_psnr_pt, copy_opt_file
)
from utils.detection import (
    CocoEvaluator, draw_box, get_coco_api_from_dataset,
    collate_fn, _get_iou_types, list_to_batch, batch_to_list
)
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
        if args.save_img and accelerator.is_local_main_process:
            img_dir = os.path.join(exp_dir, f'results_s{args.seed}', 'img')
            os.makedirs(img_dir, exist_ok=True)
            box_dir = os.path.join(exp_dir, f'results_s{args.seed}', 'box')
            os.makedirs(box_dir, exist_ok=True)
        Logging(f"Random seed: {args.seed}")

    # Create models
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    swinir.load_state_dict(torch.load(cfg.test.resume_swinir, map_location="cpu"), strict=True)
    Logging(f"Load SwinIR weight from checkpoint: {cfg.test.resume_swinir}")
    
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

    # Prepare models for testing
    swinir.eval().to(device)
    cldm.eval().to(device)
    detnet.eval().to(device)
    diffusion.to(device)    
    swinir, cldm, detnet, val_loader = accelerator.prepare(swinir, cldm, detnet, val_loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    pure_detnet = accelerator.unwrap_model(detnet)
    
    # Evaluation
    if accelerator.mixed_precision == 'fp16':
        Logging("Mixed precision is applied")
    val_psnr = []
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
        val_bs = len(val_gt_list)
        val_prompt = [cfg.test.default_prompt] * val_bs
        assert (len(val_gt_list) == 1)
        val_lq_batch = list_to_batch(val_lq_list, img_size=512, device=device)
        
        with torch.no_grad():    
            # Restoration
            val_pre_res_batch = swinir(val_lq_batch)
            val_cond = pure_cldm.prepare_condition(val_pre_res_batch, val_prompt)
            val_z = sampler.sample(
                model=cldm, device=device, steps=50, batch_size=val_bs, x_size=val_cond["c_img"].shape[1:],
                cond=val_cond, uncond=None, cfg_scale=1.0, x_T=None,
                progress=accelerator.is_local_main_process, progress_leave=False
            )
            val_res_batch = wavelet_reconstruction((pure_cldm.vae_decode(val_z) + 1) / 2, val_pre_res_batch)
            val_res_list = batch_to_list(val_res_batch, val_gt_list)
            
            # Detection
            val_pred_list, _ = detnet(val_res_list)
            val_pred_list = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in val_pred_list]
            res = {annot["image_id"]: pred for annot, pred in zip(val_annot_list, val_pred_list)}
            
            # Save images
            if args.save_img and accelerator.is_local_main_process:
                for idx, basename in enumerate(val_path_list):
                    basename = os.path.basename(basename)
                    img_name = os.path.splitext(os.path.join(img_dir, basename))[0] + ".png"
                    save_image(val_res_list[idx], img_name)
                    
                    val_pred_box = draw_box(val_res_list[idx], val_pred_list[idx],
                                            score_threshold=0.8, fontsize=0.7, split_acc=True)
                    box_name = os.path.splitext(os.path.join(box_dir, basename))[0] + ".png"
                    save_image(val_pred_box, box_name)
            
            # Calculate metrics
            val_gt, val_res = val_gt_list[0].unsqueeze(0), val_res_list[0].unsqueeze(0)
            val_psnr_batch = calculate_psnr_pt(val_res, val_gt, crop_border=0).unsqueeze(0)
            val_psnr_batch = accelerator.gather_for_metrics(val_psnr_batch)
            if accelerator.is_local_main_process:
                for v in val_psnr_batch: val_psnr += [v.item()]
            coco_evaluator.update(res)
            
        accelerator.wait_for_everyone()
        val_pbar.update(1)
    val_pbar.close()
    
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    
    if accelerator.is_local_main_process:
        avg_val_psnr = torch.tensor(val_psnr).mean().item()
        det_results = coco_evaluator.summarize(Logging=Logging)
        for tag, val in [
            ("val/psnr", avg_val_psnr),
            ("val/mAP@[0.5:0.95]", det_results["mAP@[0.5:0.95]"]),
            ("val/mAP@0.5", det_results["mAP@0.5"]),
        ]:
            Logging(f"{tag}: {val:.4f}")

    accelerator.wait_for_everyone()
    Logging("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--save-img", action='store_true')
    args = parser.parse_args()
    main(args)
