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
from utils.detection import (
    GroupedBatchSampler, CocoEvaluator, prepare_environment, prepare_batch,
    create_aspect_ratio_groups, get_coco_api_from_dataset, batch_to_list, 
    collate_fn, _get_iou_types, draw_box, suppress_stdout
)
from utils.sampler import SpacedSampler
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
        img_dir, box_dir = dirs["img"], dirs["box"]
    is_coco = True if cfg.dataset.get('is_coco') else False

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
    
    teacher_detnet = instantiate_from_config(cfg.model.detnet)  # detnet trained on HQ images
    if args.calc_fd:
        load_path = cfg.test.resume_teacher_detnet
        teacher_detnet = load_network(teacher_detnet, load_path, strict=True)
        Logging(f"Load Teacher DetectionNetwork weight from checkpoint: {load_path}")
    
    if cfg.test.resume_detnet is None:
        cfg.test.resume_detnet = os.path.join(exp_dir, 'checkpoints', 'detnet_last.pt')
    detnet = instantiate_from_config(cfg.model.detnet)
    detnet = load_network(detnet, cfg.test.resume_detnet, strict=True)
    Logging(f"Load DetectionNetwork weight from checkpoint: {cfg.test.resume_detnet}")
    
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    sampler = SpacedSampler(diffusion.betas)
    val_ts, val_N = cfg.test.start_timestep, cfg.test.num_timesteps
    val_used_timesteps = [math.floor(val_ts/val_N*i) for i in range(1, val_N+1)]
    Logging(f"Used val timesteps are specified as {val_used_timesteps}, total number of {val_N}")
    
    # setup data
    with suppress_stdout():
        val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers, shuffle=False, pin_memory=True, collate_fn=collate_fn
    )
    Logging(f"Validation dataset contains {len(val_dataset):,} images from {val_dataset.root}")

    # prepare models, evaluator, testing logs
    swinir.eval().to(device)
    cldm.eval().to(device)
    teacher_detnet.eval().to(device)
    detnet.eval().to(device)
    diffusion.to(device)    
    swinir, cldm, teacher_detnet, detnet, val_loader = \
        accelerator.prepare(swinir, cldm, teacher_detnet, detnet, val_loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    pure_detnet = accelerator.unwrap_model(detnet)
    Logging("Preparing COCO API for evaluation ..")
    with suppress_stdout():
        coco = get_coco_api_from_dataset(val_dataset)
    iou_types = _get_iou_types(pure_detnet)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    val_psnr, val_fd = [], []
    
    # Testing:
    Logging(f"Testing start...")
    val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch",
                    total=len(val_loader), leave=False, desc="Validation")
    for val_batch in val_loader:
        val_gt_list, _, _, val_lq_batch, val_annot_list, val_path_list, val_bs = prepare_batch(val_batch, device)
        val_prompt = [cfg.test.default_prompt] * val_bs
        assert (val_bs == 1)
        
        with torch.no_grad():    
            # pre-restoration
            val_pre_res_batch = val_lq_batch
            if cfg.model.pre_restoration: val_pre_res_batch = swinir(val_lq_batch)
            
            # prepare condition
            val_z_pre_res = pure_cldm.vae_encode(val_pre_res_batch * 2 - 1, sample=False)
            val_cond = dict(c_txt=pure_cldm.clip.encode(val_prompt), c_img=val_z_pre_res)
            
            # partial diffusion
            val_noise = torch.randn_like(val_z_pre_res)
            val_t = torch.tensor([val_ts] * val_bs, dtype=torch.int64).to(device)
            val_z_partial = diffusion.q_sample(x_start=val_z_pre_res, t=val_t, noise=val_noise)
            
            # short-step denoising
            val_z = sampler.manual_sample_with_timesteps(
                model=cldm, device=device, x_T=val_z_partial, steps=len(val_used_timesteps),
                used_timesteps=val_used_timesteps, batch_size=val_bs, cond=val_cond, uncond=None,
                cfg_scale=1.0, progress=accelerator.is_local_main_process, progress_leave=False
            )
            val_res_batch = wavelet_reconstruction((pure_cldm.vae_decode(val_z) + 1) / 2, val_pre_res_batch)
            val_res_list = batch_to_list(val_res_batch, val_gt_list)
            
            # detection
            val_pred_list, _ = detnet(val_res_list)
            val_pred_list = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in val_pred_list]
            res = {annot["image_id"]: pred for annot, pred in zip(val_annot_list, val_pred_list)}
            
            # calculate feature-distance
            if args.calc_fd:
                _, _, feat_gt = teacher_detnet(val_gt_list, return_feat=True)
                _, _, feat_res = teacher_detnet(val_res_list, return_feat=True)
            
            # save images
            if args.save_img and accelerator.is_local_main_process:
                for idx, basename in enumerate(val_path_list):
                    basename = os.path.basename(basename)
                    img_name = os.path.splitext(os.path.join(img_dir, basename))[0] + ".png"
                    save_image(val_res_list[idx], img_name)
                    
                    val_pred_box = draw_box(val_res_list[0], val_pred_list[0], is_coco=is_coco,
                                            score_threshold=0.8, fontsize=0.7, split_acc=True)
                    box_name = os.path.splitext(os.path.join(box_dir, basename))[0] + ".png"
                    save_image(val_pred_box, box_name)
            
            # calculate metrics
            val_gt, val_res = val_gt_list[0].unsqueeze(0), val_res_list[0].unsqueeze(0)
            val_psnr_batch = calculate_psnr_pt(val_res, val_gt, crop_border=0).unsqueeze(0)
            val_psnr_batch = accelerator.gather_for_metrics(val_psnr_batch)
            if accelerator.is_local_main_process:
                for v in val_psnr_batch: val_psnr += [v.item()]
                if args.calc_fd:
                    val_fd += [(F.l1_loss(input=feat_res['features']['0'], target=feat_gt['features']['0'], reduction="mean") * 0.5 + \
                        F.l1_loss(input=feat_res['features']['1'], target=feat_gt['features']['1'], reduction="mean") * 0.5).item()]
            coco_evaluator.update(res)
            
        accelerator.wait_for_everyone()
        val_pbar.update(1)
    val_pbar.close()
    
    coco_evaluator.synchronize_between_processes()
    with suppress_stdout():
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
