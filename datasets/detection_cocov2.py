import io
import os
import cv2
import copy
import math
import time
import torch
import torch.nn.functional as F
import random
import numpy as np

from PIL import Image
from typing import Mapping, Any, Optional, overload, Dict, List, Sequence
from torchvision import transforms
from torchvision.datasets import CocoDetection

from utils.common import instantiate_from_config
from datasets.utils import USMSharp, filter2D
from datasets.diffjpeg import DiffJPEG
from datasets.degradation import (
    random_mixed_kernels,
    circular_lowpass_kernel,
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt
)


class DegradedDetectionDatasetCocov2(CocoDetection):
    """Detection dataset with RealESRGAN degradation.
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
        # blur kernel settings of the first degradation stage
        blur_kernel_size,
        kernel_list,
        kernel_prob,
        blur_sigma,
        betag_range,
        betap_range,
        sinc_prob,
        # blur kernel settings of the second degradation stage
        blur_kernel_size2,
        kernel_list2,
        kernel_prob2,
        blur_sigma2,
        betag_range2,
        betap_range2,
        sinc_prob2,
        final_sinc_prob,
        image_set: str = 'train',
        transform: transforms = None,
        target_transform: transforms = None,
        exclude_no_annotation: bool = True,
        data_length: int = -1,
    ) -> "DegradedDetectionDatasetCocov2":
        anno_file_template = "{}_{}2017.json"
        PATHS = {
            "train": ("train2017", os.path.join("annotations", anno_file_template.format("instances", "train"))),
            "val": ("val2017", os.path.join("annotations", anno_file_template.format("instances", "val"))),
        }
        img_folder, ann_file = PATHS[image_set]
        img_folder = os.path.join(root, img_folder)
        ann_file = os.path.join(root, ann_file)
        super(DegradedDetectionDatasetCocov2, self).__init__(img_folder, ann_file, transform, target_transform)
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
        self.betag_range = betag_range
        self.betap_range = betap_range
        self.sinc_prob = sinc_prob

        self.blur_kernel_size2 = blur_kernel_size2
        self.kernel_list2 = kernel_list2
        self.kernel_prob2 = kernel_prob2
        self.blur_sigma2 = blur_sigma2
        self.betag_range2 = betag_range2
        self.betap_range2 = betap_range2
        self.sinc_prob2 = sinc_prob2

        # a final sinc filter
        self.final_sinc_prob = final_sinc_prob

        # kernel size ranges from 7 to 21
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        # TODO: kernel range is now hard-coded, should be in the configure file
        # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1
        
        self.image_set = image_set
        self.data_length = data_length
        # Number of no-annotations samples: 1021
        # Exclude samples without annotations
        if exclude_no_annotation:
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
        # -------------------------------- Load hq images -------------------------------- #
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
        img_hq = (img_gt[..., ::-1] / 255.0).astype(np.float32)
        
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None,
            )
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2,
                [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None,
            )

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # [0, 1], BGR to RGB, HWC to CHW
        img_hq = torch.from_numpy(img_hq[..., ::-1].transpose(2, 0, 1).copy()).float()
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)
        
        return img_hq, kernel, kernel2, sinc_kernel, annot, image_path
        
    def __len__(self) -> int:
        if self.data_length > len(self.ids):
            return self.data_length
        else:
            return len(self.ids)


class BatchTransform:

    @overload
    def __call__(self, batch: Any) -> Any: ...


class IdentityBatchTransform(BatchTransform):

    def __call__(self, batch: Any) -> Any:
        return batch


class RealESRGANBatchTransform(BatchTransform):

    def __init__(
        self,
        hq_key,
        extra_keys,
        use_sharpener,
        queue_size,
        resize_prob,
        resize_range,
        gray_noise_prob,
        gaussian_noise_prob,
        noise_range,
        poisson_scale_range,
        jpeg_range,
        second_blur_prob,
        stage2_scale,
        resize_prob2,
        resize_range2,
        gray_noise_prob2,
        gaussian_noise_prob2,
        noise_range2,
        poisson_scale_range2,
        jpeg_range2,
        resize_back=True,
    ):
        super().__init__()
        self.hq_key = hq_key
        self.extra_keys = extra_keys

        # resize settings for the first degradation process
        self.resize_prob = resize_prob
        self.resize_range = resize_range

        # noise settings for the first degradation process
        self.gray_noise_prob = gray_noise_prob
        self.gaussian_noise_prob = gaussian_noise_prob
        self.noise_range = noise_range
        self.poisson_scale_range = poisson_scale_range
        self.jpeg_range = jpeg_range

        self.second_blur_prob = second_blur_prob
        self.stage2_scale = stage2_scale

        # resize settings for the second degradation process
        self.resize_prob2 = resize_prob2
        self.resize_range2 = resize_range2

        # noise settings for the second degradation process
        self.gray_noise_prob2 = gray_noise_prob2
        self.gaussian_noise_prob2 = gaussian_noise_prob2
        self.noise_range2 = noise_range2
        self.poisson_scale_range2 = poisson_scale_range2
        self.jpeg_range2 = jpeg_range2

        self.use_sharpener = use_sharpener
        if self.use_sharpener:
            self.usm_sharpener = USMSharp()
        else:
            self.usm_sharpener = None
        self.queue_size = queue_size
        self.jpeger = DiffJPEG(differentiable=False)

        self.queue = {}
        self.resize_back = resize_back

    @torch.no_grad()
    def _dequeue_and_enqueue(self, values: Dict[str, torch.Tensor | List[str]]) -> Dict[str, torch.Tensor | List[str]]:
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        if len(self.queue):
            if set(values.keys()) != set(self.queue.keys()):
                raise ValueError(f"Key mismatch, input keys: {values.keys()}, queue keys: {self.queue.keys()}")
        else:
            for k, v in values.items():
                if not isinstance(v, (torch.Tensor, list)):
                    raise TypeError(f"Queue of type {type(v)} is not supported")
                if isinstance(v, list) and not isinstance(v[0], str):
                    raise TypeError("Only support queue for list of string")

                if isinstance(v, torch.Tensor):
                    size = (self.queue_size, *v.shape[1:])
                    self.queue[k] = torch.zeros(size=size, dtype=v.dtype, device=v.device)
                elif isinstance(v, list):
                    self.queue[k] = [None] * self.queue_size
            self.queue_ptr = 0

        for k, v in values.items():
            if self.queue_size % len(v) != 0:
                raise ValueError(f"Queue size {self.queue_size} should be divisible by batch size {len(v)} for key {k}")

        results = {}
        if self.queue_ptr == self.queue_size:
            # The queue is full, do dequeue and enqueue
            idx = torch.randperm(self.queue_size)
            for k, q in self.queue.items():
                v = values[k]
                b = len(v)
                if isinstance(q, torch.Tensor):
                    # Shuffle the queue
                    q_shuf = q[idx]
                    # Get front samples
                    results[k] = q_shuf[0:b, ...].clone()
                    # Update front samples
                    q_shuf[0:b, ...] = v.clone()
                    self.queue[k] = q_shuf
                else:
                    q_shuf = [q[i] for i in idx]
                    results[k] = q_shuf[0:b]
                    for i in range(b):
                        q_shuf[i] = v[i]
                    self.queue[k] = q_shuf
        else:
            # Only do enqueue
            for k, q in self.queue.items():
                v = values[k]
                b = len(v)
                if isinstance(q, torch.Tensor):
                    q[self.queue_ptr : self.queue_ptr + b, ...] = v.clone()
                else:
                    for i in range(b):
                        q[self.queue_ptr + i] = v[i]
            results = copy.deepcopy(values)
            self.queue_ptr = self.queue_ptr + b

        return results

    @torch.no_grad()
    def __call__(self, batch: Dict[str, torch.Tensor | List[str]]) -> Dict[str, torch.Tensor | List[str]]:
        hq = batch[self.hq_key]
        if self.use_sharpener:
            self.usm_sharpener.to(hq)
            hq = self.usm_sharpener(hq)
        self.jpeger.to(hq)

        kernel1 = batch["kernel1"]
        kernel2 = batch["kernel2"]
        sinc_kernel = batch["sinc_kernel"]

        ori_h, ori_w = hq.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(hq, kernel1)
        # random resize
        updown_type = random.choices(["up", "down", "keep"], self.resize_prob)[0]
        if updown_type == "up":
            scale = np.random.uniform(1, self.resize_range[1])
        elif updown_type == "down":
            scale = np.random.uniform(self.resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        if np.random.uniform() < self.gaussian_noise_prob:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.noise_range,
                clip=True,
                rounds=False,
                gray_prob=self.gray_noise_prob,
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range,
                gray_prob=self.gray_noise_prob,
                clip=True,
                rounds=False,
            )
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range)
        # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.second_blur_prob:
            out = filter2D(out, kernel2)

        # select scale of second degradation stage
        if isinstance(self.stage2_scale, Sequence):
            min_scale, max_scale = self.stage2_scale
            stage2_scale = np.random.uniform(min_scale, max_scale)
        else:
            stage2_scale = self.stage2_scale
        stage2_h, stage2_w = int(ori_h / stage2_scale), int(ori_w / stage2_scale)
        # print(f"stage2 scale = {stage2_scale}")

        # random resize
        updown_type = random.choices(["up", "down", "keep"], self.resize_prob2)[0]
        if updown_type == "up":
            scale = np.random.uniform(1, self.resize_range2[1])
        elif updown_type == "down":
            scale = np.random.uniform(self.resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, size=(int(stage2_h * scale), int(stage2_w * scale)), mode=mode)
        # add noise
        if np.random.uniform() < self.gaussian_noise_prob2:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.noise_range2,
                clip=True,
                rounds=False,
                gray_prob=self.gray_noise_prob2,
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range2,
                gray_prob=self.gray_noise_prob2,
                clip=True,
                rounds=False,
            )

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, size=(stage2_h, stage2_w), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, size=(stage2_h, stage2_w), mode=mode)
            out = filter2D(out, sinc_kernel)

        # resize back to gt_size since We are doing restoration task
        if stage2_scale != 1 and self.resize_back:
            out = F.interpolate(out, size=(ori_h, ori_w), mode="bicubic")
        # clamp and round
        lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

        batch = {"GT": hq, "LQ": lq, **{k: batch[k] for k in self.extra_keys}}
        if self.queue_size > 0:
            batch = self._dequeue_and_enqueue(batch)
        return batch
