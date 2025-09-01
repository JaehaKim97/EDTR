import cv2
import numpy as np
import random

from PIL import Image


def center_crop_arr(img_hq, image_size, img_lq=None):
    img_hq = np.array(img_hq)
    crop_y = (img_hq.shape[0] - image_size) // 2
    crop_x = (img_hq.shape[1] - image_size) // 2
    cropped_img_hq = img_hq[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    if img_lq is None:
        return cropped_img_hq
    else:
        img_lq = np.array(img_lq)
        cropped_img_lq = img_lq[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
        return cropped_img_hq, cropped_img_lq


def random_crop_arr(img_hq, image_size, img_lq=None):
    img_hq = np.array(img_hq)
    crop_y = random.randrange(img_hq.shape[0] - image_size + 1)
    crop_x = random.randrange(img_hq.shape[1] - image_size + 1)
    cropped_img_hq = img_hq[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    if img_lq is None:
        return cropped_img_hq
    else:
        img_lq = np.array(img_lq)
        cropped_img_lq = img_lq[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
        return cropped_img_hq, cropped_img_lq


# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/data/transforms.py
def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def expand2square(pil_img, background_color=(0,0,0)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
