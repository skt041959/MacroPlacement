import torch
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.diffusion import DiffusionScheduler

def test_scheduler_initialization():
    num_steps = 100
    scheduler = DiffusionScheduler(num_steps=num_steps)
    assert len(scheduler.alphas_cumprod) == num_steps
    assert torch.all(scheduler.alphas_cumprod <= 1.0)
    assert torch.all(scheduler.alphas_cumprod >= 0.0)

def test_add_noise():
    scheduler = DiffusionScheduler(num_steps=50)
    x_0 = torch.randn(10, 2)
    t = torch.tensor([25]) # Mid-point
    
    noisy_x, noise = scheduler.add_noise(x_0, t)
    
    assert noisy_x.shape == x_0.shape
    assert noise.shape == x_0.shape
    assert not torch.equal(noisy_x, x_0)

def test_add_noise_t0():
    scheduler = DiffusionScheduler(num_steps=50)
    x_0 = torch.randn(10, 2)
    t = torch.tensor([0])
    
    noisy_x, _ = scheduler.add_noise(x_0, t)
    
    # At t=0, noise should be minimal but not necessarily zero depending on alpha_0
    # But it should be very close to x_0
    assert torch.allclose(noisy_x, x_0, atol=0.1)
