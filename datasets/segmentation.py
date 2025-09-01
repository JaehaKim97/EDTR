import io
import os
import cv2
import math
import time
import torch
import random
import numpy as np

from PIL import Image
from utils.common import instantiate_from_config
from typing import Sequence, Dict, Mapping, Any, Optional
from datasets.degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression
)
from datasets.augment import augment
from datasets.utils import center_crop_arr, random_crop_arr
from torchvision import transforms
from torchvision.datasets import VOCSegmentation


class DegradedSegmentationDataset(VOCSegmentation):
    """Segmentation dataset with codeformer degradation.
    """
    def __init__(
        self,
        root: str,
        file_backend_cfg: Mapping[str, Any],
        gt_size: int,
        resize_range: Sequence[int],
        out_size: int,
        crop_type: str,
        hflip: bool,
        rotation: bool,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int],
        year: str = '2012',
        image_set: str = 'train',
        download: bool = False,
        transform: transforms = None,
        target_transform: transforms = None,
        data_length: int = -1,
    ) -> "DegradedSegmentationDataset":
        super(DegradedSegmentationDataset, self).__init__(root, year, image_set, download, transform, target_transform)
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.gt_size = gt_size
        self.resize_range = resize_range
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.hflip = hflip
        self.rotation = rotation
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range
        self.data_length = data_length
        
    def load_items(self, image_path: str, mask_path: str, max_retry: int=5) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask = Image.open(mask_path)
        
        if self.resize_range is not None:
            r = random.uniform(*self.resize_range)
        else:
            r = 1.0
        w, h = image.size
        if w >= h:
            image = image.resize((int(self.gt_size * w / h * r), int(self.gt_size * r)), Image.BICUBIC)
            mask = mask.resize((int(self.gt_size * w / h * r), int(self.gt_size * r)), Image.NEAREST)
        else:
            image = image.resize((int(self.gt_size * r), int(self.gt_size * h / w * r)), Image.BICUBIC)
            mask = mask.resize((int(self.gt_size * r), int(self.gt_size * h / w * r)), Image.NEAREST)
        
        image, mask = np.array(image), np.array(mask)
        min_size = min(mask.shape)
        if (self.out_size is not None) and min_size < self.out_size:
            ow, oh = mask.shape
            padh = self.out_size - oh if oh < self.out_size else 0
            padw = self.out_size - ow if ow < self.out_size else 0
            image = np.pad(image, ((0, padw), (0, padh), (0,0)), mode='constant', constant_values=0)
            mask = np.pad(mask, ((0, padw), (0, padh)), mode='constant', constant_values=255)
        
        if self.crop_type != "none":
            if image.shape[0] == self.out_size and image.shape[1] == self.out_size:
                image, mask = np.array(image), np.array(mask)
            else:
                if self.crop_type == "center":
                    image = center_crop_arr(image, self.out_size)
                    mask = center_crop_arr(mask, self.out_size)
                elif self.crop_type == "random":
                    image, crop_pos = random_crop_arr(image, self.out_size, return_params=True)
                    mask = random_crop_arr(mask, self.out_size, crop_pos=crop_pos)

        image, mask = augment([image, mask], self.hflip, self.rotation)
        
        # hwc, rgb, 0,255, uint8
        return image, mask

    def __getitem__(self, index: int, max_retry: int=5) -> Dict[str, torch.Tensor]:
        index = index % len(self.images)
        # load gt image
        img_gt = None
        while img_gt is None:
            # load meta file
            gt_path, mask_path = self.images[index], self.masks[index]
            img_gt, mask = self.load_items(gt_path, mask_path)
            if img_gt is None:
                print(f"filed to load {gt_path}, try another image")
                index = random.randint(0, len(self) - 1)
        
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        img_gt = (img_gt[..., ::-1] / 255.0).astype(np.float32)
        h, w, _ = img_gt.shape
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        if self.blur_kernel_size is not None:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                noise_range=None
            )
            img_lq = cv2.filter2D(img_gt, -1, kernel)
        else:
            img_lq = img_gt
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if self.noise_range is not None:
            img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        
        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # BGR to RGB, [0, 1]
        gt = img_gt[..., ::-1].astype(np.float32)
        # BGR to RGB, [0, 1]
        lq = img_lq[..., ::-1].astype(np.float32)
        
        return gt, lq, mask, gt_path
        
    def __len__(self) -> int:
        if self.data_length > len(self.images):
            return self.data_length
        else:
            return len(self.images)


class PairedSegmentationDataset(VOCSegmentation):
    """Segmentation dataset from existing image pairs (HQ, LQ).
    """
    def __init__(
        self,
        root: str,
        path: str,
        file_backend_cfg: Mapping[str, Any],
        year: str = '2012',
        image_set: str = 'val',
        center_crop: bool = False,
        download: bool = False,
        transform: transforms = None,
        target_transform: transforms = None,
    ) -> "PairedSegmentationDataset":
        super(PairedSegmentationDataset, self).__init__(root, year, image_set, download, transform, target_transform)
        self.path = path
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.center_crop = center_crop
                
    def load_items(self, gt_path: str, lq_path: str, mask_path: str, max_retry: int=5) -> Optional[np.ndarray]:
        gt_bytes, lq_bytes = None, None
        while (gt_bytes is None) or (lq_bytes is None):
            if max_retry == 0:
                return None
            gt_bytes = self.file_backend.get(gt_path)
            lq_bytes = self.file_backend.get(lq_path)
            max_retry -= 1
            if (gt_bytes is None) or (lq_bytes is None):
                time.sleep(0.5)
        image_gt = Image.open(io.BytesIO(gt_bytes)).convert("RGB")
        image_lq = Image.open(io.BytesIO(lq_bytes)).convert("RGB")
        mask = Image.open(mask_path)
        mask = mask.resize((image_gt.size[0], image_gt.size[1]), Image.NEAREST)
        
        if self.center_crop:
            image_gt = center_crop_arr(image_gt, 512)
            image_lq = center_crop_arr(image_lq, 512)
            mask = center_crop_arr(mask, 512)
        
        image_gt, image_lq, mask = np.array(image_gt), np.array(image_lq), np.array(mask)
        
        # hwc, rgb, 0,255, uint8
        return image_gt, image_lq, mask
    
    def __getitem__(self, index: int, max_retry: int=5) -> Dict[str, torch.Tensor]:
        # load gt, lq images
        img_gt, img_lq = None, None
        while (img_gt is None) or (img_lq is None):
            # load meta file
            img_path, mask_path = self.images[index], self.masks[index]
            gt_path = img_path.replace("JPEGImages", os.path.join(self.path, "gt")).replace('.jpg', '.png')
            lq_path = img_path.replace("JPEGImages", os.path.join(self.path, "lq")).replace('.jpg', '.png')
            img_gt, img_lq, mask = self.load_items(gt_path, lq_path, mask_path)
            if (img_gt is None) or (img_lq is None):
                print(f"failed to load {gt_path} and {lq_path}, try another image")
                index = random.randint(0, len(self) - 1)
        
        # Shape: (h, w, c); channel order: RGB; image range: [0, 1], float32.
        img_gt = (img_gt / 255.0).astype(np.float32)
        img_lq = (img_lq / 255.0).astype(np.float32)

        return img_gt, img_lq, mask, gt_path

    def __len__(self) -> int:
        return len(self.images)
