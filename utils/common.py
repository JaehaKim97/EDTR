import os
import cv2
import numpy as np
import torch
import logging
import importlib

from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
from torch import Tensor
from torch.nn import functional as F
from torch.hub import download_url_to_file, get_dir
from typing import Mapping, Any, Tuple, Callable
from urllib.parse import urlparse
from torchvision.models import get_model
from torchvision.transforms.functional import normalize


def get_obj_from_str(string: str, reload: bool=False) -> Any:
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: Mapping[str, Any]) -> Any:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def wavelet_blur(image: Tensor, radius: int):
    """
    Apply wavelet blur to the input tensor.
    """
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    # apply convolution
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output


def wavelet_decomposition(image: Tensor, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq

    return high_freq, low_freq


def wavelet_reconstruction(content_feat:Tensor, style_feat:Tensor):
    """
    Apply wavelet decomposition, so that the content will have the same color as the style.
    """
    # calculate the wavelet decomposition of the content feature
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    # calculate the wavelet decomposition of the style feature
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    # reconstruct the content feature with the style's high frequency
    return content_high_freq + style_low_freq


# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/download_util.py/
def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


def sliding_windows(h: int, w: int, tile_size: int, tile_stride: int) -> Tuple[int, int, int, int]:
    hi_list = list(range(0, h - tile_size + 1, tile_stride))
    if (h - tile_size) % tile_stride != 0:
        hi_list.append(h - tile_size)
    
    wi_list = list(range(0, w - tile_size + 1, tile_stride))
    if (w - tile_size) % tile_stride != 0:
        wi_list.append(w - tile_size)
    
    coords = []
    for hi in hi_list:
        for wi in wi_list:
            coords.append((hi, hi + tile_size, wi, wi + tile_size))
    return coords


# https://github.com/csslc/CCSR/blob/main/model/q_sampler.py#L503
def gaussian_weights(tile_width: int, tile_height: int) -> np.ndarray:
    """Generates a gaussian mask of weights for tile contributions"""
    latent_width = tile_width
    latent_height = tile_height
    var = 0.01
    midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
    x_probs = [
        np.exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / np.sqrt(2 * np.pi * var)
        for x in range(latent_width)]
    midpoint = latent_height / 2
    y_probs = [
        np.exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / np.sqrt(2 * np.pi * var)
        for y in range(latent_height)]
    weights = np.outer(y_probs, x_probs)
    return weights


COUNT_VRAM = bool(os.environ.get("COUNT_VRAM", False))

def count_vram_usage(func: Callable) -> Callable:
    if not COUNT_VRAM:
        return func
    
    def wrapper(*args, **kwargs):
        peak_before = torch.cuda.max_memory_allocated() / (1024 ** 3)
        ret = func(*args, **kwargs)
        torch.cuda.synchronize()
        peak_after = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"VRAM peak before {func.__name__}: {peak_before:.5f} GB, after: {peak_after:.5f} GB")
        return ret
    return wrapper


def log_txt_as_img(wh, xc):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        # font = ImageFont.truetype('font/DejaVuSans.ttf', size=size)
        font = ImageFont.load_default()
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


# https://github.com/XPixelGroup/BasicSR/blob/033cd6896d898fdd3dcda32e3102a792efa1b8f4/basicsr/utils/color_util.py#L186
def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB images to YCbCr images (PyTorch version).

    It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    Args:
        img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
         y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img


# https://github.com/XPixelGroup/BasicSR/blob/033cd6896d898fdd3dcda32e3102a792efa1b8f4/basicsr/metrics/psnr_ssim.py#L52
def calculate_psnr_pt(img, img2, crop_border, test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img - img2)**2, dim=[1, 2, 3])
    return 10. * torch.log10(1. / (mse + 1e-8))


def calculate_lpips_pt(img, img2, net_lpips, crop_border=8, img_range=1.0, **kwargs):
    """Computes the PSNR (Peak-Signal-Noise-Ratio) in batch"""
        
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    
    # norm to [-1, 1]
    img = normalize(img, mean, std)
    img2 = normalize(img2, mean, std)

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
        
    lpips = net_lpips(img, img2).squeeze(1,2,3)  # batch-wise lpips
    return lpips


def _print_different_keys_loading(crt_net, load_net, strict=True):
    """Print keys with different name or different size when loading models.

    1. Print keys with different names.
    2. If strict=False, print the same key but with different tensor size.
        It also ignore these keys with different sizes (not load).

    Args:
        crt_net (torch model): Current network.
        load_net (dict): Loaded network.
        strict (bool): Whether strictly loaded. Default: True.
    """
    crt_net = crt_net.state_dict()
    crt_net_keys = set(crt_net.keys())
    load_net_keys = set(load_net.keys())

    if crt_net_keys != load_net_keys:
        print('Current net - loaded net:')
        for v in sorted(list(crt_net_keys - load_net_keys)):
            print(f'  {v}')
        print('Loaded net - current net:')
        for v in sorted(list(load_net_keys - crt_net_keys)):
            print(f'  {v}')

    # check the size for the same keys
    if not strict:
        common_keys = crt_net_keys & load_net_keys
        for k in common_keys:
            if crt_net[k].size() != load_net[k].size():
                print(f'Size different, ignore [{k}]: crt_net: '
                      f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                load_net[k + '.ignore'] = load_net.pop(k)


def load_network(crt_net, load_path, strict):
    """Load network.

    Args:
        load_path (str): The path of networks to be loaded.
        net (nn.Module): Network.
        strict (bool): Whether strictly loaded.
        param_key (str): The parameter key of loaded network. If set to
            None, use the root 'path'.
            Default: 'params'.
    """
    if os.path.exists(load_path):
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        _print_different_keys_loading(crt_net, load_net, strict)
        crt_net.load_state_dict(load_net, strict=strict)
    else:
        try:
            load_net = get_model('ResNet18', weights=load_path, num_classes=1000,).state_dict()
            _print_different_keys_loading(crt_net, load_net, strict)
            crt_net.load_state_dict(load_net, strict=strict)
        except:
            raise NotImplementedError(f'{load_path} is not valid model path!')
        
    return crt_net


def copy_opt_file(opt_file, experiments_root):
    # copy the yml file to the experiment root
    import sys
    import time
    from shutil import copyfile
    
    cmd = 'accelerate launch ' + ' '.join(sys.argv)
    filename = os.path.join(experiments_root, opt_file)
    os.makedirs(os.path.join(experiments_root, os.path.dirname(opt_file)), exist_ok=True)
    copyfile(opt_file, filename)

    with open(filename, 'r+') as f:
        lines = f.readlines()
        lines.insert(0, f'# GENERATE TIME: {time.asctime()}\n# CMD:\n# {cmd}\n\n')
        f.seek(0)
        f.writelines(lines)


def set_logger(file_name, exp_dir, logger_name):
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(f'{exp_dir}/{logger_name}')
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    streamHandler.setLevel(level=logging.INFO)  # Save and print
    fileHandler.setLevel(level=logging.DEBUG)  # Only save
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    return logger


def print_attn_type(Logging):
    # Attention type:
    from model.config import SDP_IS_AVAILABLE, XFORMERS_IS_AVAILBLE
    if SDP_IS_AVAILABLE:
        Logging("Using sdp attention as default")
    elif XFORMERS_IS_AVAILBLE:
        Logging("Using xformers attention as default")   
    return
