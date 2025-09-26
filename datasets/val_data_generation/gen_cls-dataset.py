import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from argparse import ArgumentParser
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from utils.common import instantiate_from_config, set_logger
from torchvision.utils import save_image


def main(args) -> None:
    # Setup accelerator
    cfg = OmegaConf.load(args.config)
    set_seed(231)
    
    # Setup an experiment folder
    exp_dir = cfg.val.exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment directory created at {exp_dir}")
    logger = set_logger(__name__, exp_dir, logger_name="logger.log")
    
    # setup data
    dataset = instantiate_from_config(cfg.dataset.val)
    loader = DataLoader(
        dataset=dataset, batch_size=cfg.val.batch_size,
        num_workers=cfg.val.num_workers,
        shuffle=False, drop_last=False
    )
    logger.info(f"Validation dataset contains {len(dataset):,} images from {dataset.root}")

    # Making dataset:
    for gt, lq, _, gt_path in tqdm(loader):
        gt = rearrange(gt, "b h w c -> b c h w")
        lq = rearrange(lq, "b h w c -> b c h w")
        for idx, img_name in enumerate(gt_path):
            # save gt
            img_name = os.path.splitext(os.path.join(exp_dir, 'gt', *gt_path[idx].split('/')[-2:]))[0] + ".png"
            os.makedirs(os.path.dirname(img_name), exist_ok=True)
            save_image(gt[idx], img_name)
            
            # save lq
            img_name = os.path.splitext(os.path.join(exp_dir, 'lq', *gt_path[idx].split('/')[-2:]))[0] + ".png"
            os.makedirs(os.path.dirname(img_name), exist_ok=True)
            save_image(lq[idx], img_name)
    logger.info("done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
