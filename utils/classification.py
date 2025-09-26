import os
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
            dirs["img"] = os.path.join(exp_dir, f'results_s{seed}', 'img')
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


def calculate_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res