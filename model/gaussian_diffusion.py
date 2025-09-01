import numpy as np
import torch

from numpy.polynomial.chebyshev import Chebyshev
from torch import nn
from typing import Tuple


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
            np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            np.arange(n_timestep + 1, dtype=np.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = np.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "sqrt":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas


def extract_into_tensor(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int]) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class Diffusion(nn.Module):

    def __init__(
        self,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        parameterization="eps",
    ):
        super().__init__()
        self.num_timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.cosine_s = cosine_s
        assert parameterization in ["eps", "x0", "v"], "currently only supporting 'eps' and 'x0' and 'v'"
        self.parameterization = parameterization
        self.loss_type = loss_type
        
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                   cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        self.betas = betas
        self.register("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        self.register("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
    
    def register(self, name: str, value: np.ndarray) -> None:
        self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

    def q_sample(self, x_start, t, noise):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        
    def get_v(self, x, noise, t):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss
    
    def predict_xstart_from_eps(self, x_t, t, eps):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
        
    def p_losses(self, model, x_start, t, cond, return_predicted_x_start=False):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = model(x_noisy, t, cond)

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()
        
        loss_simple = self.get_loss(model_output, target, mean=False).mean()
        
        if return_predicted_x_start:
            predicted_x_start = self.predict_xstart_from_eps(x_noisy, t, model_output)
            return loss_simple, predicted_x_start
        else:
            return loss_simple

    def modified_p_losses(self, model, x0, x_start, t, cond, return_predicted_x0=False):
        # only supports eps-mode
        assert (self.parameterization == "eps")
        
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = model(x_noisy, t, cond)

        target = noise + (x_start - x0) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x0.shape)
        
        loss_simple = self.get_loss(model_output, target, mean=False).mean()
        
        if return_predicted_x0:
            predicted_x0 = self.predict_xstart_from_eps(x_noisy, t, model_output)
            return loss_simple, predicted_x0
        else:
            return loss_simple
        
    def reverse(self, model, t, x0, cond, noise=None, x_noisy=None):
        # only supports eps-mode
        assert (self.parameterization == "eps")
        
        if noise is None:
            noise = torch.randn_like(x0)
        if x_noisy is None:
            x_noisy = self.q_sample(x_start=x0, t=t, noise=noise)
        
        model_output = model(x_noisy, t, cond)
        outputs = dict(
            x_noisy=x_noisy,
            x_pred=self.predict_xstart_from_eps(x_noisy, t, model_output),
            model_output=model_output
        )
        
        return outputs
    
    def reverse_woSD(self, model, t, x0, cond):
        # only supports eps-mode
        assert (self.parameterization == "eps")
        
        model_output = model(x0, t, cond, woSD=True) + x0
        
        outputs = dict(x_pred=model_output)
        
        return outputs
