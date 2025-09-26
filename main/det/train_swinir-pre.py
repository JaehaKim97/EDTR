import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
import utils.filter_warning

import torch
from tqdm import tqdm
from model import SwinIR
from torch.nn import functional as F
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
from torchvision.utils import make_grid, save_image


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
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    if cfg.train.resume_swinir:
        swinir.load_state_dict(torch.load(cfg.train.resume_swinir, map_location="cpu"), strict=True)
        Logging(f"Load SwinIR weight from checkpoint: {cfg.train.resume_swinir}")
    else:
        Logging("Initialize SwinIR from scratch")
    
    # setup optimizer
    opt = torch.optim.AdamW(
        swinir.parameters(), lr=cfg.train.learning_rate,
        weight_decay=0
    )
    
    # setup scheduler
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.train.train_steps,
        eta_min=1e-7
    )
    
    # setup data
    if cfg.dataset.get('use_gt'): Logging(f"Using ground-truth image!")
    with suppress_stdout():
        dataset = instantiate_from_config(cfg.dataset.train)
    train_sampler = torch.utils.data.RandomSampler(dataset)
    group_ids = create_aspect_ratio_groups(
        dataset, k=cfg.train.aspect_ratio_group_factor, Logging=Logging
    )
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
    
    # prepare models, training logs
    swinir.train().to(device)
    swinir, opt, sch, loader, val_loader = accelerator.prepare(swinir, opt, sch, loader, val_loader)
    pure_swinir = accelerator.unwrap_model(swinir)
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
            _, _, gt_batch, lq_batch, _, _, _ = prepare_batch(batch, device, batch_transform)
            
            with accelerator.autocast():
                res_batch = swinir(lq_batch)
                loss = F.l1_loss(input=res_batch, target=gt_batch, reduction="mean") * 255.0
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step(), sch.step()
            
            accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(loss.item())
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch:04d}, Steps: {global_step:07d}, Loss: {loss.item():.6f}")

            # log training loss, learning rate
            if global_step % cfg.train.log_every == 0 or (args.debug):
                avg_loss = accelerator.gather(torch.tensor(step_loss, device=device).unsqueeze(0)).mean().item()
                step_loss.clear()
                loss_summary = f"[{global_step:05d}/{max_steps:05d}] Training loss: {avg_loss:.4f}"
                Logging(loss_summary, print=False)
                if accelerator.is_local_main_process:
                    writer.add_scalar("train/loss_step", avg_loss, global_step)
                    writer.add_scalar("train/learning_rate", opt.param_groups[0]['lr'], global_step)

            # save checkpoint
            if global_step % cfg.train.ckpt_every == 0 or (args.debug):
                if accelerator.is_local_main_process:
                    torch.save(pure_swinir.state_dict(), f"{ckpt_dir}/{global_step:07d}.pt")

            # save images
            if global_step % cfg.train.image_every == 0 or global_step == 1 or (args.debug):
                swinir.eval()
                N = 12
                log_gt, log_lq = gt_batch[:N], lq_batch[:N]
                with torch.no_grad():
                    log_res = swinir(log_lq)
                if accelerator.is_local_main_process:
                    for tag, image in [
                        ("image/restored", log_res),
                        ("image/gt", log_gt),
                        ("image/lq", log_lq),
                    ]:
                        grid_image = make_grid(image, nrow=4)
                        writer.add_image(tag, grid_image, global_step)
                        img_name = '{}_{:06d}.png'.format(os.path.basename(tag), global_step)
                        save_image(grid_image, os.path.join(img_dir, img_name))
                swinir.train()

            # evaluation
            if global_step % cfg.val.val_every == 0 or (args.debug):
                swinir.eval()
                val_psnr = []
                val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch",
                                total=len(val_loader), leave=False, desc="Validation")
                for val_batch in val_loader:
                    val_gt_list, val_lq_list, _, _, _, _, val_bs = prepare_batch(val_batch, device)
                    assert (val_bs == 1)
                    
                    val_gt = val_gt_list[0].unsqueeze(0)
                    val_lq = val_lq_list[0].unsqueeze(0)
                    
                    h, w = val_lq.shape[2:]
                    ph, pw = 512 - h, 512 - w
                    val_lq_padded = F.pad(val_lq, pad=(0, pw, 0, ph))
                    
                    with torch.no_grad():
                        val_res = swinir(val_lq_padded)[:,:,:h,:w]
                        _val_psnr = calculate_psnr_pt(val_res, val_gt, crop_border=0).unsqueeze(0)
                        _val_psnr = accelerator.gather_for_metrics(_val_psnr)
                        
                        if accelerator.is_local_main_process:
                            for v in _val_psnr:
                                val_psnr += [v.item()]
                        accelerator.wait_for_everyone()
                    val_pbar.update(1)
                val_pbar.close()
                
                if accelerator.is_local_main_process:
                    avg_val_psnr = torch.tensor(val_psnr).mean().item()
                    for tag, val in [
                        ("val/psnr", avg_val_psnr)
                    ]:
                        Logging(f"{tag}: {val:.4f}")
                        writer.add_scalar(tag, val, global_step)
                swinir.train()
            
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break
        
        pbar.close()
        epoch += 1

    # save the last model weight
    if accelerator.is_local_main_process:
        torch.save(pure_swinir.state_dict(), f"{ckpt_dir}/swinir_last.pt")
        Logging("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(args)
