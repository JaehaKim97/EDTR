from typing import Optional, Tuple, Dict, List

import torch
from torch import nn
import numpy as np
from tqdm import tqdm

from model.gaussian_diffusion import extract_into_tensor
from model import ControlLDM
from utils.common import make_tiled_fn


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/respace.py
def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedSampler(nn.Module):
    """
    Implementation for spaced sampling schedule proposed in IDDPM. This class is designed
    for sampling ControlLDM.
    
    https://arxiv.org/pdf/2102.09672.pdf
    """
    
    def __init__(self, betas: np.ndarray) -> "SpacedSampler":
        super().__init__()
        self.num_timesteps = len(betas)
        self.original_betas = betas
        self.original_alphas_cumprod = np.cumprod(1.0 - betas, axis=0)
        self.context = {}

    def register(self, name: str, value: np.ndarray) -> None:
        self.register_buffer(name, torch.tensor(value, dtype=torch.float32))
    
    def make_schedule(self, num_steps: int, used_timesteps=None) -> None:
        # calcualte betas for spaced sampling
        # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/respace.py
        if used_timesteps is None:
            used_timesteps = space_timesteps(self.num_timesteps, str(num_steps))
        betas = []
        last_alpha_cumprod = 1.0
        for i, alpha_cumprod in enumerate(self.original_alphas_cumprod):
            if i in used_timesteps:
                # marginal distribution is the same as q(x_{S_t}|x_0)
                betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        assert len(betas) == num_steps
        self.timesteps = np.array(sorted(list(used_timesteps)), dtype=np.int32) # e.g. [0, 10, 20, ...]

        betas = np.array(betas, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # print(f"sampler sqrt_alphas_cumprod: {np.sqrt(alphas_cumprod)[-1]}")
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        if num_steps == 1:
            posterior_log_variance_clipped = np.array([-10.0])
        else:
            posterior_log_variance_clipped = np.log(
                np.append(posterior_variance[1], posterior_variance[1:])
            )
        posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - alphas_cumprod)
        )

        self.register("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
        self.register("posterior_variance", posterior_variance)
        self.register("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register("posterior_mean_coef1", posterior_mean_coef1)
        self.register("posterior_mean_coef2", posterior_mean_coef2)

    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Implement the posterior distribution q(x_{t-1}|x_t, x_0).
        
        Args:
            x_start (torch.Tensor): The predicted images (NCHW) in timestep `t`.
            x_t (torch.Tensor): The sampled intermediate variables (NCHW) of timestep `t`.
            t (torch.Tensor): Timestep (N) of `x_t`. `t` serves as an index to get 
                parameters for each timestep.
        
        Returns:
            posterior_mean (torch.Tensor): Mean of the posterior distribution.
            posterior_variance (torch.Tensor): Variance of the posterior distribution.
            posterior_log_variance_clipped (torch.Tensor): Log variance of the posterior distribution.
        """
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _predict_xstart_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def predict_noise(
        self,
        model: ControlLDM,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float
    ) -> torch.Tensor:
        if uncond is None or cfg_scale == 1.:
            model_output = model(x, t, cond)
        else:
            # apply classifier-free guidance
            model_cond = model(x, t, cond)
            model_uncond = model(x, t, uncond)
            model_output = model_uncond + cfg_scale * (model_cond - model_uncond)
        return model_output
    
    @torch.no_grad()
    def p_sample(
        self,
        model: ControlLDM,
        x: torch.Tensor,
        t: torch.Tensor,
        index: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float,
    ) -> torch.Tensor:
        eps = self.predict_noise(model, x, t, cond, uncond, cfg_scale)
        pred_x0 = self._predict_xstart_from_eps(x, index, eps)
        
        model_mean, model_variance, _ = self.q_posterior_mean_variance(pred_x0, x, index)
        noise = torch.randn_like(x)
        nonzero_mask = (
            (index != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        x_prev = model_mean + nonzero_mask * torch.sqrt(model_variance) * noise
        return x_prev, pred_x0

    @torch.no_grad()
    def sample(
        self,
        model: ControlLDM,
        device: str,
        steps: int,
        batch_size: int,
        x_size: Tuple[int],
        cond: Dict[str, torch.Tensor],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
        tiled: bool=False,
        tile_size: int=-1,
        tile_stride: int=-1,
        x_T: Optional[torch.Tensor]=None,
        progress: bool=True,
        progress_leave: bool=True,
        return_intermediates: bool=False
    ) -> torch.Tensor:
        self.make_schedule(steps)
        self.to(device)
        if tiled:
            forward = model.forward
            model.forward = make_tiled_fn(
                lambda x_tile, t, cond, hi, hi_end, wi, wi_end: (
                    forward(
                        x_tile,
                        t,
                        {
                            "c_txt": cond["c_txt"],
                            "c_img": cond["c_img"][..., hi:hi_end, wi:wi_end],
                        },
                    )
                ),
                tile_size,
                tile_stride,
            )
        if x_T is None:
            # TODO: not convert to float32, may trigger an error
            img = torch.randn((batch_size, *x_size), device=device)
        else:
            img = x_T
        timesteps = np.flip(self.timesteps) # [1000, 950, 900, ...]
        total_steps = len(self.timesteps)
        iterator = tqdm(timesteps, total=total_steps, leave=progress_leave, disable=not progress)
        intermediates = []
        for i, step in enumerate(iterator):
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
            index = torch.full_like(ts, fill_value=total_steps - i - 1)
            img, pred_x0 = self.p_sample(
                model, img, ts, index, cond, uncond, cfg_scale
            )
            if return_intermediates:
                intermediates.append(pred_x0)
            desc = "Spaced Sampler"
            iterator.set_description(desc)
        if return_intermediates:
            return img, intermediates
        else:
            return img
          
    @torch.no_grad()
    def manual_sample_with_timesteps(
        self,
        model: ControlLDM,
        device: str,
        x_T: Optional[torch.Tensor],
        steps: int,
        used_timesteps: List[int],
        batch_size: int,
        cond: Dict[str, torch.Tensor],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
        tiled: bool=False,
        tile_size: int=-1,
        tile_stride: int=-1,
        progress: bool=True,
        progress_leave: bool=True,
        return_intermediates: bool=False,
    ) -> torch.Tensor:
        self.make_schedule(steps, used_timesteps)
        self.to(device)
        if tiled:
            forward = model.forward
            model.forward = make_tiled_fn(
                lambda x_tile, t, cond, hi, hi_end, wi, wi_end: (
                    forward(
                        x_tile,
                        t,
                        {
                            "c_txt": cond["c_txt"],
                            "c_img": cond["c_img"][..., hi:hi_end, wi:wi_end],
                        },
                    )
                ),
                tile_size,
                tile_stride,
            )
        img = x_T
        
        timesteps = np.flip(self.timesteps)
        total_steps = len(self.timesteps)
        iterator = tqdm(timesteps, total=total_steps, leave=progress_leave, disable=not progress)
        intermediates = []
        for i, step in enumerate(iterator):
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
            index = torch.full_like(ts, fill_value=total_steps - i - 1)
            img, pred_x0 = self.p_sample(
                model, img, ts, index, cond, uncond, cfg_scale
            )
            if return_intermediates:
                intermediates.append(pred_x0)
            desc = "Spaced Sampler"
            iterator.set_description(desc)
        if return_intermediates:
            return img, intermediates
        else:
            return img
