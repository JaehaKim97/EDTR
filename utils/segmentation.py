import os
import numpy as np
import torch
from utils.common import copy_opt_file, print_attn_type, Logger
from accelerate.utils import set_seed


def prepare_environment(name, cfg, args, accelerator, is_oracle=False):
    dirs = dict()
    if cfg.get("train"):
        exp_dir = cfg.train.exp_dir
        seed = cfg.train.seed
        dirs["exp"] = exp_dir
        dirs["ckpt"] = os.path.join(exp_dir, "checkpoints")
        dirs["img"] = os.path.join(exp_dir, "images")
        os.makedirs(exp_dir, exist_ok=True)
        Logging = Logger(name, exp_dir, accelerator, logger_name="logger.log")
    elif cfg.get("test"):
        exp_dir = cfg.test.exp_dir
        seed = args.seed
        dirs["exp"] = cfg.test.exp_dir        
        if args.save_img:
            if is_oracle:
                dirs["pred_mask"] = os.path.join(exp_dir, f'results_s{seed}', 'pred_mask')
                dirs["gt_mask"] = os.path.join(exp_dir, f'results_s{seed}', 'gt_mask')
            else:
                dirs["img"] = os.path.join(exp_dir, f'results_s{seed}', 'img')
                dirs["mask"] = os.path.join(exp_dir, f'results_s{seed}', 'mask')
        os.makedirs(exp_dir, exist_ok=True)
        Logging = Logger(name, exp_dir, accelerator, logger_name="logger_test.log")
    else:
        raise NotImplementedError("Error: mode is not clear; training or test?")
    
    if accelerator.is_local_main_process:
        for k in dirs.keys():
            os.makedirs(dirs[k], exist_ok=True)
    Logging(f"Experiment directory created at {exp_dir}")
    
    set_seed(seed)
    Logging(f"Random seed: {seed}")
    
    copy_opt_file(args.config, exp_dir)
    print_attn_type(Logging=Logging)
    
    if accelerator.mixed_precision == 'fp16':
        Logging("Mixed precision is applied")
        
    return dirs, Logging


@torch.inference_mode()
def convert2color(mask):
    template = torch.zeros_like(mask).expand(3, *mask.shape).to(torch.float64)
    max_rgb = 255.0
    
    # Color maps for 21 classes:
    color_list = [
        [158, 184, 217],
        [124, 147, 195],
        [162, 87, 114],
        [97, 163, 186],
        [255, 255, 221],
        [210, 222, 50],
        [162, 197, 121],
        [162, 197, 121],
        [0, 66, 90], #[252, 239, 145], 
        [31, 138, 112],
        [191, 219, 56],
        [252, 115, 0],
        [131, 162, 255],
        [180, 189, 255],
        [255, 227, 187],
        [255, 210, 143],
        [251, 236, 178],
        [248, 189, 235],
        [82, 114, 242],
        [7, 37, 65],
        [188, 122, 249],
    ]
    
    dontcare = (mask==max_rgb)
    if dontcare.sum() != 0:
        color = [0, 0, 0]  # color_list[0]
        color = np.array(color) / max_rgb
        template[:, dontcare] = torch.Tensor(color).to(template.dtype).to(template.device).view(3,1).repeat(1, dontcare.sum())
        
    for idx in range(21):
        mask_idx = (mask==idx)
        if mask_idx.sum() != 0:
            color = color_list[idx]
            color = np.array(color) / max_rgb
            template[:, mask_idx] = torch.Tensor(color).to(template.dtype).to(template.device).view(3,1).repeat(1,mask_idx.sum())
    
    template = template.permute(1,0,2,3)
    
    return template


def calculate_mat(pred, target, n):
    k = (pred >= 0) & (pred < n)
    inds = n * pred[k].to(torch.int64) + target[k]
    return torch.bincount(inds, minlength=n**2).reshape(n, n)


def compute_iou(mat):
    h = mat.float()
    iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
    return iu
