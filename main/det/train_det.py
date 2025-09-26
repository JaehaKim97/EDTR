import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
import utils.filter_warning

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.common import (
    instantiate_from_config, load_network,
    calculate_psnr_pt, wavelet_reconstruction
)
from utils.detection import (
    GroupedBatchSampler, CocoEvaluator, prepare_environment, prepare_batch,
    create_aspect_ratio_groups, get_coco_api_from_dataset, batch_to_list, 
    collate_fn, _get_iou_types, draw_box, suppress_stdout
)
from argparse import ArgumentParser
from omegaconf import OmegaConf
from accelerate import Accelerator, DataLoaderConfiguration
from torchvision.utils import save_image


def main(args) -> None:
    # setup environment
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cfg = OmegaConf.load(args.config)
    accelerator = Accelerator(dataloader_config=DataLoaderConfiguration(split_batches=True),
                              mixed_precision=cfg.train.precision)
    device = accelerator.device
    dirs, Logging = prepare_environment(__name__, cfg, args, accelerator)
    exp_dir, ckpt_dir, img_dir = dirs["exp"], dirs["ckpt"], dirs["img"]
    is_coco = True if cfg.dataset.get('is_coco') else False
    
    # create and load models
    detnet = instantiate_from_config(cfg.model.detnet)
    if cfg.train.resume_detnet:
        detnet = load_network(detnet, cfg.train.resume_detnet, strict=cfg.train.strict_load)
        Logging(f"Load DetectionNetwork weight from checkpoint: {cfg.train.resume_detnet}, strict: {cfg.train.strict_load}")
    else:
        Logging("Initialize DetectionNetwork from scratch")
    
    # setup optimizer and scheduler
    opt = torch.optim.SGD(
        detnet.parameters(), lr=cfg.train.learning_rate,
        momentum=0.9, weight_decay=1e-4
    )
    warmup_iters = 1000 if is_coco else 500
    sch_warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=(1/warmup_iters), total_iters=warmup_iters
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=(cfg.train.train_steps - warmup_iters),
        eta_min=1e-7
    )
    
    # setup data
    if cfg.dataset.get('use_gt'): Logging(f"Using ground-truth image!")
    with suppress_stdout():
        dataset = instantiate_from_config(cfg.dataset.train)
    train_sampler = torch.utils.data.RandomSampler(dataset)
    group_ids = create_aspect_ratio_groups(dataset, k=cfg.train.aspect_ratio_group_factor, Logging=Logging)
    batch_sampler = GroupedBatchSampler(train_sampler, group_ids, cfg.train.batch_size)
    loader = DataLoader(
        dataset=dataset, batch_sampler=batch_sampler,
        num_workers=cfg.train.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    batch_transform = None
    if cfg.dataset.get("batch_transform"):
        batch_transform = instantiate_from_config(cfg.dataset.batch_transform)
    Logging(f"Training dataset contains {len(dataset):,} images from {dataset.root}")
    
    if cfg.val.batch_size == -1:
        num_gpus = accelerator.state.num_processes
        cfg.val.batch_size = num_gpus
        Logging(f"Validation batch size is automatically set to the number of available GPUs: {num_gpus}")
    with suppress_stdout():
        val_dataset = instantiate_from_config(cfg.dataset.val)    
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=cfg.val.batch_size, shuffle=False,
        num_workers=cfg.val.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    Logging(f"Validation dataset contains {len(val_dataset):,} images from {val_dataset.root}")

    # prepare models, evaluator, training logs
    detnet.train().to(device)
    detnet, opt, sch, loader, val_loader = accelerator.prepare(detnet, opt, sch, loader, val_loader)
    pure_detnet = accelerator.unwrap_model(detnet)
    Logging("Preparing COCO API for evaluation...")
    with suppress_stdout():
        coco = get_coco_api_from_dataset(val_dataset)
    iou_types = _get_iou_types(pure_detnet)
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    epoch = 0
    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
    
    # Training:
    Logging(f"Training for {max_steps} steps...")
    while global_step < max_steps:
        pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch", total=len(loader))
        for batch in loader:
            gt_list, lq_list, _, _, annot_list, _, _ = prepare_batch(batch, device, batch_transform)
            
            # for idx in range(len(gt_list)):
            #     gt_box = draw_box(gt_list[idx], annot_list[idx], is_coco=is_coco)
            #     save_image(gt_box, os.path.join(f"gt_{global_step:06d}_{idx:02d}.png"))
            #     lq_box = draw_box(lq_list[idx], annot_list[idx], is_coco=is_coco)
            #     save_image(lq_box, os.path.join(f"lq_{global_step:06d}_{idx:02d}.png"))
            
            # input image type
            inp_list = gt_list if cfg.dataset.get('use_gt') else lq_list
            
            with accelerator.autocast():
                _, loss_dict = detnet(inp_list, annot_list)
                loss = sum(loss_value for loss_value in loss_dict.values())

            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            if global_step < warmup_iters:
                sch_warmup.step()
            else:
                sch.step()
            accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(loss.item())
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch:04d}, Steps: {global_step:07d}, Loss: {loss.item():.6f}")

            # log training loss, learning rate
            if global_step % cfg.train.log_every == 0 or (args.debug):
                avg_loss = accelerator.gather(torch.tensor(step_loss, device=device).unsqueeze(0)).mean().item()
                step_loss.clear()
                
                if accelerator.is_local_main_process:
                    loss_summary = f"[{global_step:05d}/{max_steps:05d}] Training loss: {avg_loss:.4f}"
                    Logging(loss_summary, print=False)
                
                    writer.add_scalar("train/loss_step", avg_loss, global_step)
                    writer.add_scalar("train/learning_rate", opt.param_groups[0]['lr'], global_step)
                    
            # save checkpoint
            if global_step % cfg.train.ckpt_every == 0 or (args.debug):
                if accelerator.is_local_main_process:
                    checkpoint = pure_detnet.state_dict()
                    ckpt_path = f"{ckpt_dir}/detnet_{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)
            
            # evaluation
            if global_step % cfg.val.val_every == 0 or (args.debug):
                detnet.eval()
                coco_evaluator = CocoEvaluator(coco, iou_types)
                val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch",
                                total=len(val_loader), leave=False, desc="Validation")
                for idx, val_batch in enumerate(val_loader):
                    val_gt_list, val_lq_list, _, _, val_annot_list, _, val_bs = prepare_batch(val_batch, device)
                    assert (val_bs == 1)
                    
                    # input image type
                    val_inp_list = val_gt_list if cfg.dataset.get('use_gt') else val_lq_list
                    
                    # detection
                    with torch.no_grad():
                        val_pred_list, _ = detnet(val_inp_list)
                        val_pred_list = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in val_pred_list]
                    
                        # save images
                        if idx < 5 and accelerator.is_local_main_process:
                            val_gt_box = draw_box(val_gt_list[0], val_annot_list[0], is_coco=is_coco)
                            val_pred_box = draw_box(val_inp_list[0], val_pred_list[0], is_coco=is_coco)
                            save_image(val_gt_box, os.path.join(img_dir, f"val_gt_{global_step:06d}_{idx:02d}.png"))
                            save_image(val_pred_box, os.path.join(img_dir, f"val_pred_{global_step:06d}_{idx:02d}.png"))
                        
                        # evaluation:
                        res = {annot["image_id"]: pred for annot, pred in zip(val_annot_list, val_pred_list)}
                        coco_evaluator.update(res)
                    val_pbar.update(1)
                val_pbar.close()    
                
                coco_evaluator.synchronize_between_processes()
                with suppress_stdout():
                    coco_evaluator.accumulate()
                
                if accelerator.is_local_main_process:
                    det_results = coco_evaluator.summarize(Logging=Logging)
                    for tag, val in [
                        ("val/mAP@[0.5:0.95]", det_results["mAP@[0.5:0.95]"]),
                        ("val/mAP@0.5", det_results["mAP@0.5"]),
                    ]:
                        Logging(f"{tag}: {val:.1f}")
                        writer.add_scalar(tag, val, global_step)
                detnet.train()
            
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break
        
        pbar.close()
        epoch += 1

    # save the last model weight
    if accelerator.is_local_main_process:            
        torch.save(pure_detnet.state_dict(), f"{ckpt_dir}/detnet_last.pt")
        Logging("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(args)
