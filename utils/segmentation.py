import numpy as np
import torch


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
