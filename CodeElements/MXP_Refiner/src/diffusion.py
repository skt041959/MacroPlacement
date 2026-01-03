import torch
import numpy as np
from config import Config

class DiffusionScheduler:
    def __init__(self, num_steps=None, beta_start=None, beta_end=None):
        self.num_steps = num_steps or Config.DIFFUSION_TIMESTEPS
        self.beta_start = beta_start or Config.BETA_START
        self.beta_end = beta_end or Config.BETA_END
        
        # Linear beta schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps)
        
        # Derived values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # For forward process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x_0, t, noise=None):
        """
        Forward process: q(x_t | x_0)
        x_0: [N, 2] original coordinates
        t: [B] or [1] timesteps
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise, noise
