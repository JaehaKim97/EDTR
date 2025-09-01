import os
import sys
import torch
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from tqdm import tqdm
from einops import rearrange
from argparse import ArgumentParser
from torch.nn import functional as F
from omegaconf import OmegaConf
from accelerate import Accelerator
from utils.common import instantiate_from_config, load_network, copy_opt_file, set_logger, print_attn_type
from utils.segmentation import convert2color, calculate_mat, compute_iou
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter


def main(args) -> None:
    # Setup accelerator
    cfg = OmegaConf.load(args.config)
    if args.debug: cfg.train.batch_size, cfg.val.batch_size = 8, 1
    accelerator = Accelerator(split_batches=True, mixed_precision=cfg.train.precision)
    set_seed(cfg.train.seed)
    device = accelerator.device
    
    def Logging(text, print=True):
        if accelerator.is_local_main_process:
            if print:
                logger.info(text)
            else:
                logger.debug(text)

    # Set up the experiment folder and logger
    if accelerator.is_local_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        img_dir = os.path.join(exp_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")
        logger = set_logger(__name__, exp_dir, logger_name="logger.log")
        copy_opt_file(args.config, exp_dir)
        print_attn_type(Logging=Logging)
        Logging(f"Random seed: {cfg.train.seed}")

    # Create models
    segnet = instantiate_from_config(cfg.model.segnet)
    if cfg.train.resume_segnet:
        segnet = load_network(segnet, cfg.train.resume_segnet, strict=cfg.train.strict_load)
        Logging(f"Load SegmentationNetwork weight from checkpoint: {cfg.train.resume_segnet}, strict: {cfg.train.strict_load}")
    else:
        Logging("Initialize SegmentationNetwork from scratch")
    
    # Setup optimizer:
    opt = torch.optim.SGD(
        segnet.parameters(), lr=cfg.train.learning_rate,
        momentum=0.9, weight_decay=1e-6
    )
    
    # Setup scheduler:
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.train.train_steps,
        eta_min=1e-7
    )
    
    # Setup data
    if cfg.dataset.get('use_gt'):
        Logging(f"Using ground-truth image!")
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True, drop_last=True
    )
    Logging(f"Training dataset contains {len(dataset):,} images from {dataset.root}")
    
    if cfg.val.batch_size == -1:
        num_gpus = accelerator.state.num_processes
        cfg.val.batch_size = num_gpus
        Logging(f"Validation batch size is automatically set to the number of available GPUs: {num_gpus}")
    val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=cfg.val.batch_size,
        num_workers=cfg.val.num_workers,
        shuffle=False, drop_last=False
    )
    Logging(f"Validation dataset contains {len(val_dataset):,} images from {val_dataset.root}")

    # Prepare models for training
    segnet.train().to(device)
    segnet, opt, sch, loader, val_loader = accelerator.prepare(segnet, opt, sch, loader, val_loader)
    pure_segnet = accelerator.unwrap_model(segnet)

    # Define variables related to training
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    epoch = 0
    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
    
    # Training
    if accelerator.mixed_precision == 'fp16':
        Logging("Mixed precision is applied")    
    Logging(f"Training for {max_steps} steps...")
    
    while global_step < max_steps:
        pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch", total=len(loader))
        for gt, lq, mask, _ in loader:
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float().to(device)
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float().to(device)
            mask = mask.long()
            
            # Input image type
            inp = gt if cfg.dataset.get('use_gt') else lq
            
            pred = segnet(inp, normalize=True)
            loss = F.cross_entropy(pred["out"], mask, ignore_index=255)
            # loss += 0.5 * F.cross_entropy(pred["aux"], mask, ignore_index=255)

            opt.zero_grad()
            accelerator.backward(loss)
            opt.step(), sch.step()
            accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(loss.item())
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch:04d}, Steps: {global_step:07d}, Loss: {loss.item():.6f}")

            # Log training loss, learning rate
            if global_step % cfg.train.log_every == 0 or (args.debug):
                avg_loss = accelerator.gather(torch.tensor(step_loss, device=device).unsqueeze(0)).mean().item()
                step_loss.clear()
                loss_summary = f"[{global_step:05d}/{max_steps:05d}] Training loss: {avg_loss:.4f}"
                Logging(loss_summary, print=False)
                if accelerator.is_local_main_process:
                    writer.add_scalar("train/loss_step", avg_loss, global_step)
                    writer.add_scalar("train/learning_rate", opt.param_groups[0]['lr'], global_step)

            # Save checkpoint
            if global_step % cfg.train.ckpt_every == 0 or (args.debug):
                if accelerator.is_local_main_process:
                    checkpoint = pure_segnet.state_dict()
                    ckpt_path = f"{ckpt_dir}/segnet_{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)
            
            # Save images
            if global_step % cfg.train.image_every == 0 or global_step == 1 or (args.debug):
                with torch.no_grad():
                    N = 4
                    log_gt, log_lq, log_mask, log_pred = gt[:N], lq[:N], convert2color(mask[:N]), convert2color(pred["out"][:N].argmax(1))
                    
                    if accelerator.is_local_main_process:
                        for tag, image in [
                            ("image/gt", log_gt), ("image/lq", log_lq),
                            ("image/mask", log_mask), ("image/pred", log_pred),
                        ]:
                            grid_image = make_grid(image, nrow=4)
                            writer.add_image(tag, grid_image, global_step)
                            img_name = '{}_{:06d}.png'.format(os.path.basename(tag), global_step)
                            save_image(grid_image, os.path.join(img_dir, img_name))
                        del grid_image
                    del log_gt, log_lq, log_mask, log_pred
                    torch.cuda.empty_cache()
            
            # Evaluation
            if global_step % cfg.val.val_every == 0 or (args.debug):
                segnet.eval()
                n_classes = 21
                confmat = torch.zeros((n_classes, n_classes), device=device)
                val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch",
                                total=len(val_loader), leave=False, desc="Validation")
                for val_gt, val_lq, val_mask, _ in val_loader:
                    val_gt = rearrange(val_gt, "b h w c -> b c h w").contiguous().float().to(device)
                    val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float().to(device)
                    val_mask = val_mask.long()
                    
                    # Input image type
                    val_inp = val_gt if cfg.dataset.get('use_gt') else val_lq
                    
                    with torch.no_grad():
                        val_pred = segnet(val_inp, normalize=True)
                    val_mat = calculate_mat(val_mask.flatten(), val_pred['out'].argmax(1).flatten(), n=n_classes).unsqueeze(0)
                    val_mat = accelerator.gather_for_metrics(val_mat)
                    
                    if accelerator.is_local_main_process:
                        confmat += val_mat.sum(0)
                    accelerator.wait_for_everyone()
                    val_pbar.update(1)
                val_pbar.close()
                
                if accelerator.is_local_main_process:
                    miou = compute_iou(confmat).mean().item() * 100
                    for tag, val in [
                        ("val/miou", miou),
                    ]:
                        Logging(f"{tag}: {val:.4f}")
                        writer.add_scalar(tag, val, global_step)
                segnet.train()
            
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break
        
        pbar.close()
        epoch += 1

    # Save the last model weight
    if accelerator.is_local_main_process:            
        torch.save(pure_segnet.state_dict(), f"{ckpt_dir}/segnet_last.pt")
        Logging("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(args)
