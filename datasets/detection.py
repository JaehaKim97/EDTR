import io
import os
import cv2
import xml.etree.ElementTree as ET
import math
import time
import torch
import random
import numpy as np

from PIL import Image
from glob import glob
from typing import Sequence, Dict, Mapping, Any, Optional
from torchvision import transforms
from torchvision.datasets import VOCDetection

from utils.common import instantiate_from_config
from datasets.utils import center_crop_arr, random_crop_arr, get_label2id, convert2coco
from datasets.degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression
)


class DegradedDetectionDataset(VOCDetection):
    """Detection dataset with codeformer degradation.
    """
    def __init__(
        self,
        root: str,
        file_backend_cfg: Mapping[str, Any],
        gt_size: int,
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
    ) -> "DegradedDetectionDataset":
        super(DegradedDetectionDataset, self).__init__(root, year, image_set, download, transform, target_transform)
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.gt_size = gt_size
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
        self.label2id = get_label2id('./datasets/voc_labels.txt')
        self.data_length = data_length
        
    def load_items(self, image_path: str, annot_path: str, max_retry: int=5) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        annot = self.parse_voc_xml(ET.parse(annot_path).getroot())
        height, width = image.shape[:2]
        
        # flip augmentation
        if self.hflip and (torch.rand(1) < 0.5):
            image = cv2.flip(image, 1, image)
            for item in annot['annotation']['object']:
                xmin, xmax = int(item['bndbox']["xmin"]), int(item['bndbox']["xmax"])
                item['bndbox']["xmin"] = str(max(width - xmax, 1))  # gurantee minimum value of 1
                item['bndbox']["xmax"] = str(width - xmin)
        
        # image resizing
        if height >= width:
            scale_factor = self.gt_size / height
            image = cv2.resize(image, dsize=(int(width * scale_factor), self.gt_size), interpolation=cv2.INTER_CUBIC)
        else:
            scale_factor = self.gt_size / width
            image = cv2.resize(image, dsize=(self.gt_size, int(height * scale_factor)), interpolation=cv2.INTER_CUBIC)
        height, width = image.shape[:2]
        for item in annot['annotation']['object']:
            xmin, xmax = int(item['bndbox']["xmin"]), int(item['bndbox']["xmax"])
            ymin, ymax = int(item['bndbox']["ymin"]), int(item['bndbox']["ymax"])
            item['bndbox']["xmin"] = str(max(int(xmin * scale_factor), 1))
            item['bndbox']["xmax"] = str(min(int(xmax * scale_factor), width))
            item['bndbox']["ymin"] = str(max(int(ymin * scale_factor), 1))
            item['bndbox']["ymax"] = str(min(int(ymax * scale_factor), height))
        
        # image cropping
        if self.crop_type != "none":
            if height == self.out_size and width == self.out_size:
                pass
            else:
                if self.crop_type == "center":
                    image, crop_pos = center_crop_arr(image, self.out_size, return_params=True)
                elif self.crop_type == "random":
                    image, crop_pos = random_crop_arr(image, self.out_size, return_params=True)
                new_y0, new_x0 = crop_pos
                new_obj = []
                for item in annot['annotation']['object']:
                    xmin, xmax = int(item['bndbox']["xmin"]), int(item['bndbox']["xmax"])
                    ymin, ymax = int(item['bndbox']["ymin"]), int(item['bndbox']["ymax"])
                    if (xmax > new_x0) and (ymax > new_y0):
                        xmin, xmax = max(xmin - new_x0, 1), min(xmax - new_x0, self.out_size)
                        ymin, ymax = max(ymin - new_y0, 1), min(ymax - new_y0, self.out_size)
                        threshold = 15
                        if (xmax > xmin + threshold) and (ymax > ymin + threshold): 
                            item['bndbox']["xmin"], item['bndbox']["xmax"] = str(xmin), str(xmax)
                            item['bndbox']["ymin"], item['bndbox']["ymax"] = str(ymin), str(ymax)
                            new_obj.append(item.copy())
                annot['annotation']['object'] = new_obj
        
        # convert to coco style
        annot = convert2coco(annot, self.label2id)
        
        # hwc, rgb, 0,255, uint8
        return image, annot

    def __getitem__(self, index: int, max_retry: int=5) -> Dict[str, torch.Tensor]:
        index = index % len(self.images)
        # load gt image
        img_gt, annot_length = None, 0
        while (img_gt is None) or (annot_length == 0):
            # load meta file
            gt_path, annot_path = self.images[index], self.annotations[index]
            img_gt, annot = self.load_items(gt_path, annot_path)
            annot_length = len(annot['boxes'])
            if (img_gt is None) or (annot_length == 0):
                print(f"failed to load {gt_path}, try another image")
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
        gt = torch.Tensor(img_gt[..., ::-1].astype(np.float32))
        lq = torch.Tensor(img_lq[..., ::-1].astype(np.float32))
        annot = {k: torch.Tensor(v) if isinstance(v, list) else v for k, v in annot.items()}
        
        return gt, lq, annot, gt_path
        
    def __len__(self) -> int:
        if self.data_length > len(self.images):
            return self.data_length
        else:
            return len(self.images)


class PairedDetectionDataset(VOCDetection):
    """Detection dataset from existing image pairs (HQ, LQ).
    """
    def __init__(
        self,
        root: str,
        path: str,
        file_backend_cfg: Mapping[str, Any],
        gt_size: int,
        year: str = '2012',
        image_set: str = 'val',
        download: bool = False,
        transform: transforms = None,
        target_transform: transforms = None,
    ) -> "PairedDetectionDataset":
        super(PairedDetectionDataset, self).__init__(root, year, image_set, download, transform, target_transform)
        self.path = path
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.gt_size = gt_size
        self.label2id = get_label2id('./datasets/voc_labels.txt')
        
    def load_items(self, gt_path: str, lq_path: str, annot_path: str, max_retry: int=5) -> Optional[np.ndarray]:
        gt_bytes, lq_bytes = None, None
        while (gt_bytes is None) or (lq_bytes is None):
            if max_retry == 0:
                return None
            gt_bytes = self.file_backend.get(gt_path)
            lq_bytes = self.file_backend.get(lq_path)
            max_retry -= 1
            if (gt_bytes is None) or (lq_bytes is None):
                time.sleep(0.5)
        image_gt = np.array(Image.open(io.BytesIO(gt_bytes)).convert("RGB"))
        image_lq = np.array(Image.open(io.BytesIO(lq_bytes)).convert("RGB"))
        annot = self.parse_voc_xml(ET.parse(annot_path).getroot())
        
        height, width = int(annot['annotation']['size']['height']), int(annot['annotation']['size']['width'])
        if height >= width:
            scale_factor = self.gt_size / height
            height, width = self.gt_size, int(width * scale_factor)
        else:
            scale_factor = self.gt_size / width
            height, width = int(height * scale_factor), self.gt_size
        
        assert (image_gt.shape[:2] == (height, width))
        
        for item in annot['annotation']['object']:
            xmin, xmax = item['bndbox']["xmin"], item['bndbox']["xmax"]
            ymin, ymax = item['bndbox']["ymin"], item['bndbox']["ymax"]
            item['bndbox']["xmin"] = str(max(int(int(xmin) * scale_factor), 1))
            item['bndbox']["xmax"] = str(min(int(int(xmax) * scale_factor), width))
            item['bndbox']["ymin"] = str(max(int(int(ymin) * scale_factor), 1))
            item['bndbox']["ymax"] = str(min(int(int(ymax) * scale_factor), height))
        
        # convert to coco style
        annot = convert2coco(annot, self.label2id)
        
        # hwc, rgb, 0,255, uint8
        return image_gt, image_lq, annot
    
    def __getitem__(self, index: int, max_retry: int=5) -> Dict[str, torch.Tensor]:
        # load gt, lq images
        img_gt, img_lq = None, None
        while (img_gt is None) or (img_lq is None):
            # load meta file
            img_path, annot_path = self.images[index], self.annotations[index]
            gt_path = img_path.replace("JPEGImages", os.path.join(self.path, "gt")).replace('.jpg', '.png')
            lq_path = img_path.replace("JPEGImages", os.path.join(self.path, "lq")).replace('.jpg', '.png')
            img_gt, img_lq, annot = self.load_items(gt_path, lq_path, annot_path)
            if (img_gt is None) or (img_lq is None):
                print(f"failed to load {gt_path} and {lq_path}, try another image")
                index = random.randint(0, len(self) - 1)
        
        # Shape: (h, w, c); channel order: RGB; image range: [0, 1], float32.
        img_gt = torch.Tensor((img_gt / 255.0).astype(np.float32))
        img_lq = torch.Tensor((img_lq / 255.0).astype(np.float32))
        annot = {k: torch.Tensor(v) if isinstance(v, list) else v for k, v in annot.items()}

        return img_gt, img_lq, annot, gt_path

    def __len__(self) -> int:
        return len(self.images)


class RealworldDetectionDataset():
    def __init__(
        self,
        root: str,
        file_backend_cfg: Mapping[str, Any],
        gt_size: int,
    ) -> "RealworldDetectionDataset":
        self.root = root
        self.image_paths = glob(os.path.join(root, '*.png'))
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.gt_size = gt_size
        
    def load_image(self, image_path: str, max_retry: int=5) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        height, width = image.shape[:2]
        
        # image resizing
        if height >= width:
            scale_factor = self.gt_size / height
            image = cv2.resize(image, dsize=(int(width * scale_factor), self.gt_size), interpolation=cv2.INTER_CUBIC)
        else:
            scale_factor = self.gt_size / width
            image = cv2.resize(image, dsize=(self.gt_size, int(height * scale_factor)), interpolation=cv2.INTER_CUBIC)
        height, width = image.shape[:2]
        
        # hwc, rgb, 0,255, uint8
        return image
    
    def __getitem__(self, index: int, max_retry: int=5) -> Dict[str, torch.Tensor]:
        # load img
        img = None
        while (img is None):
            # load meta file
            img_path = self.image_paths[index]
            img = self.load_image(img_path)
            if (img is None):
                print(f"failed to load {img_path}, try another image")
                index = random.randint(0, len(self) - 1)
        
        # Shape: (h, w, c); channel order: RGB; image range: [0, 1], float32.
        img = torch.Tensor((img / 255.0).astype(np.float32))

        return img, img_path

    def __len__(self) -> int:
        return len(self.image_paths)
