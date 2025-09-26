import io
import os
import cv2
import math
import time
import torch
import random
import numpy as np

from PIL import Image
from glob import glob
from typing import Sequence, Dict, Mapping, Any, Optional
from torchvision import transforms
from torchvision.datasets import CocoDetection

from utils.common import instantiate_from_config
from datasets.utils import center_crop_arr, random_crop_arr, get_label2id, convert2coco
from datasets.degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression
)


class DegradedDetectionDatasetCoco(CocoDetection):
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
        image_set: str = 'train',
        transform: transforms = None,
        target_transform: transforms = None,
        data_length: int = -1,
    ) -> "DegradedDetectionDatasetCoco":
        anno_file_template = "{}_{}2017.json"
        PATHS = {
            "train": ("train2017", os.path.join("annotations", anno_file_template.format("instances", "train"))),
            "val": ("val2017", os.path.join("annotations", anno_file_template.format("instances", "val"))),
        }
        img_folder, ann_file = PATHS[image_set]
        img_folder = os.path.join(root, img_folder)
        ann_file = os.path.join(root, ann_file)
        super(DegradedDetectionDatasetCoco, self).__init__(img_folder, ann_file, transform, target_transform)
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
        self.image_set = image_set
        self.data_length = data_length
        # Number of no-annotations samples: 1021
        # Exclude samples without annotations
        self.ids = [id for id in self.ids if len(self.coco.loadAnns(self.coco.getAnnIds(id))) > 0]
    
    def load_items(self, id: int, max_retry: int=5) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_path = os.path.join(self.root, self.coco.loadImgs(id)[0]["file_name"])
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        height, width = image.shape[:2]
        
        annot = self.coco.loadAnns(self.coco.getAnnIds(id))
        annot = [obj for obj in annot if obj["iscrowd"] == 0]
        # convert annotation format
        # list to dict, (x1, y1, x2, y2) format
        if len(annot) > 0:
            boxes = [obj["bbox"] for obj in annot]
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(min=0, max=width)
            boxes[:, 1::2].clamp_(min=0, max=height)
            classes = [obj["category_id"] for obj in annot]
            classes = torch.tensor(classes, dtype=torch.int64)
            area = torch.tensor([obj["area"] for obj in annot])
            iscrowd = torch.tensor([obj["iscrowd"] for obj in annot])
            
            # flip augmentation
            if self.hflip and (torch.rand(1) < 0.5):
                image = cv2.flip(image, 1, image)
                xmin, xmax = boxes[:,0].clone(), boxes[:,2].clone()
                boxes[:,0] = torch.maximum(width - xmax, torch.tensor(1.0))  # gurantee minimum value of 1
                boxes[:,2] = width - xmin
            
            # image resizing
            if height >= width:
                scale_factor = self.gt_size / height
                image = cv2.resize(image, dsize=(int(width * scale_factor), self.gt_size), interpolation=cv2.INTER_CUBIC)
            else:
                scale_factor = self.gt_size / width
                image = cv2.resize(image, dsize=(self.gt_size, int(height * scale_factor)), interpolation=cv2.INTER_CUBIC)
            height, width = image.shape[:2]
            xmin, xmax = boxes[:,0].clone(), boxes[:,2].clone()
            ymin, ymax = boxes[:,1].clone(), boxes[:,3].clone()
            boxes[:,0] = torch.maximum(xmin * scale_factor, torch.tensor(1.0))
            boxes[:,2] = torch.minimum(xmax * scale_factor, torch.tensor(width))
            boxes[:,1] = torch.maximum(ymin * scale_factor, torch.tensor(1.0))
            boxes[:,3] = torch.minimum(ymax * scale_factor, torch.tensor(height))
            
            # image cropping is not supported
            if self.crop_type != "none":
                pass
            
            # keep only valid labels
            keep = (boxes[:, 3] > boxes[:, 1] + 1) & (boxes[:, 2] > boxes[:, 0] + 1)
            new_annot = {}
            new_annot["image_id"] = annot[0]["image_id"]
            new_annot["boxes"] = boxes[keep]
            new_annot["labels"] = classes[keep]
            new_annot["area"] = area[keep]
            new_annot["iscrowd"] = iscrowd[keep]
            annot = new_annot
        
        # hwc, rgb, 0,255, uint8
        return image, annot, image_path
        
    def __getitem__(self, index: int) -> tuple[Any, Any]:
        id = self.ids[index]
        # loag gt image
        img_gt, annot_length = None, 0
        while (img_gt is None) or (annot_length == 0) and (self.image_set == "train"):
            # load meta file
            img_gt, annot, image_path = self.load_items(id)
            annot_length = len(annot)
            if (img_gt is None) or (annot_length == 0):
                index = random.randint(0, len(self) - 1)
                id = self.ids[index]
        
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
        if len(annot) > 0:
            annot = {k: torch.Tensor(v) if isinstance(v, list) else v for k, v in annot.items()}

        return gt, lq, annot, image_path
        
    def __len__(self) -> int:
        if self.data_length > len(self.ids):
            return self.data_length
        else:
            return len(self.ids)


class PairedDetectionDatasetCoco(CocoDetection):
    """Detection dataset from existing image pairs (HQ, LQ).
    """
    def __init__(
        self,
        root: str,
        path: str,
        file_backend_cfg: Mapping[str, Any],
        gt_size: int,
        transform: transforms = None,
        target_transform: transforms = None,
    ) -> "PairedDetectionDatasetCoco":
        img_folder = os.path.join(root, "val2017-deg/gt", )
        ann_file = os.path.join(root, "annotations/instances_val2017.json")
        super(PairedDetectionDatasetCoco, self).__init__(img_folder, ann_file, transform, target_transform)
        self.path = path
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.gt_size = gt_size
        # Number of no-annotations samples: 48
        # Exclude samples without annotations
        self.ids = [id for id in self.ids if len(self.coco.loadAnns(self.coco.getAnnIds(id))) > 0]
    
    def load_items(self, id: int, max_retry: int=5) -> Optional[np.ndarray]:
        gt_bytes, lq_bytes = None, None
        while (gt_bytes is None) or (lq_bytes is None):
            if max_retry == 0:
                return None
            gt_path = os.path.join(self.root, self.coco.loadImgs(id)[0]["file_name"])
            gt_path = os.path.splitext(gt_path)[0] + ".png"
            gt_bytes = self.file_backend.get(gt_path)
            lq_path = gt_path.replace("val2017-deg/gt", self.path)
            lq_bytes = self.file_backend.get(lq_path)
            max_retry -= 1
            if (gt_bytes is None) or (lq_bytes is None):
                time.sleep(0.5)
        image_gt = np.array(Image.open(io.BytesIO(gt_bytes)).convert("RGB"))
        image_lq = np.array(Image.open(io.BytesIO(lq_bytes)).convert("RGB"))
        
        annot = self.coco.loadAnns(self.coco.getAnnIds(id))
        annot = [obj for obj in annot if obj["iscrowd"] == 0]
        # convert annotation format
        # list to dict, (x1, y1, x2, y2) format
        if len(annot) > 0:
            img_info = self.coco.loadImgs(annot[0]["image_id"])[0]
            height, width = img_info["height"], img_info["width"]
            
            boxes = [obj["bbox"] for obj in annot]
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(min=0, max=width)
            boxes[:, 1::2].clamp_(min=0, max=height)
            classes = [obj["category_id"] for obj in annot]
            classes = torch.tensor(classes, dtype=torch.int64)
            area = torch.tensor([obj["area"] for obj in annot])
            iscrowd = torch.tensor([obj["iscrowd"] for obj in annot])
            
            # image resizing
            if height >= width:
                scale_factor = self.gt_size / height
                height, width = self.gt_size, int(width * scale_factor)
            else:
                scale_factor = self.gt_size / width
                height, width = int(height * scale_factor), self.gt_size
            
            assert (image_gt.shape[:2] == (height, width))
            
            xmin, xmax = boxes[:,0].clone(), boxes[:,2].clone()
            ymin, ymax = boxes[:,1].clone(), boxes[:,3].clone()
            boxes[:,0] = torch.maximum(xmin * scale_factor, torch.tensor(1.0))
            boxes[:,2] = torch.minimum(xmax * scale_factor, torch.tensor(width))
            boxes[:,1] = torch.maximum(ymin * scale_factor, torch.tensor(1.0))
            boxes[:,3] = torch.minimum(ymax * scale_factor, torch.tensor(height))
            
            # keep only valid labels
            keep = (boxes[:, 3] > boxes[:, 1] + 1) & (boxes[:, 2] > boxes[:, 0] + 1)
            new_annot = {}
            new_annot["image_id"] = annot[0]["image_id"]
            new_annot["boxes"] = boxes[keep]
            new_annot["labels"] = classes[keep]
            new_annot["area"] = area[keep]
            new_annot["iscrowd"] = iscrowd[keep]
            annot = new_annot
        
        # hwc, rgb, 0,255, uint8
        return image_gt, image_lq, annot, gt_path
        
    def __getitem__(self, index: int) -> tuple[Any, Any]:
        id = self.ids[index]
        # loag gt image
        img_gt, img_lq = None, None
        while (img_gt is None) or (img_lq is None):
            # load meta file
            img_gt, img_lq, annot, gt_path = self.load_items(id)
            if (img_gt is None) or (img_lq is None):
                print(f"failed to load {gt_path}")
                raise NotImplementedError
        
        # Shape: (h, w, c); channel order: RGB; image range: [0, 1], float32.
        img_gt = torch.Tensor((img_gt / 255.0).astype(np.float32))
        img_lq = torch.Tensor((img_lq / 255.0).astype(np.float32))
        if len(annot) > 0:
            annot = {k: torch.Tensor(v) if isinstance(v, list) else v for k, v in annot.items()}

        return img_gt, img_lq, annot, gt_path
        
    def __len__(self) -> int:
        return len(self.ids)
