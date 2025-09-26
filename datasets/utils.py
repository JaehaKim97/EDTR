import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import random

from typing import Dict


def center_crop_arr(pil_image, image_size, return_params=False):
    arr = np.array(pil_image)
    
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    
    if return_params:
        crop_pos = (crop_y, crop_x)
        return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size], crop_pos
    else:
        return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, crop_pos=None, return_params=False):
    arr = np.array(pil_image)
    if crop_pos is not None:
        crop_y, crop_x = crop_pos
    else:
        crop_y = random.randrange(arr.shape[0] - image_size + 1)
        crop_x = random.randrange(arr.shape[1] - image_size + 1)
    
    if return_params:
        crop_pos = (crop_y, crop_x)
        return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size], crop_pos
    else:
        return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def get_label2id(labels_path: str) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, 'r') as f:
        labels_str = f.read().split()
    labels_ids = list(range(1, len(labels_str)+1))
    return dict(zip(labels_str, labels_ids))


def convert2coco(obj, label2id):
    ann = {'boxes': [], 'labels': [], 'image_id':[], 'area': [], 'iscrowd': []}
    ann['image_id'] = os.path.splitext(obj['annotation']['filename'])[0]
    for idx in range(len(obj['annotation']['object'])):
        each_obj = obj['annotation']['object'][idx]
        label = each_obj['name']
        assert label in label2id, f"Error: {label} is not in label2id !"
        category_id = label2id[label]
        bndbox = each_obj['bndbox']
        xmin = int(float(bndbox['xmin'])) - 1
        ymin = int(float(bndbox['ymin'])) - 1
        xmax = int(float(bndbox['xmax']))
        ymax = int(float(bndbox['ymax']))
        assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
        o_width = xmax - xmin
        o_height = ymax - ymin
        ann['boxes'].append([xmin, ymin, xmin+o_width, ymin+o_height])
        ann['labels'].append(category_id),
        ann['area'].append(o_width * o_height)
        ann['iscrowd'].append(0)
    return ann


# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/img_process_util.py
def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode="reflect")
    else:
        raise ValueError("Wrong kernel size")

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        # img: torch.Tensor
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


class USMSharp(torch.nn.Module):

    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer("kernel", kernel)

    def forward(self, img, weight=0.5, threshold=10):
        blur = filter2D(img, self.kernel)
        residual = img - blur

        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = filter2D(mask, self.kernel)
        sharp = img + weight * residual
        sharp = torch.clip(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img
