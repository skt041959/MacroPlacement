import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.diffusion import DiffusionScheduler

def check_scheduler():
    scheduler = DiffusionScheduler(num_steps=100, beta_start=1e-4, beta_end=0.02)
    
    print(f"Betas: {scheduler.betas[:5]} ... {scheduler.betas[-5:]}")
    print(f"Alphas Cumprod: {scheduler.alphas_cumprod[:5]} ... {scheduler.alphas_cumprod[-5:]}")
    print(f"Sqrt 1-AlphaCumprod: {scheduler.sqrt_one_minus_alphas_cumprod[:5]} ... {scheduler.sqrt_one_minus_alphas_cumprod[-5:]}")
    
    # Check t=0
    t0_scale = scheduler.sqrt_one_minus_alphas_cumprod[0]
    print(f"t=0 noise scale: {t0_scale.item()}")
    
    # Check t=99
    t99_scale = scheduler.sqrt_one_minus_alphas_cumprod[99]
    print(f"t=99 noise scale: {t99_scale.item()}")

if __name__ == "__main__":
    check_scheduler()
