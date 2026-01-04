import torch
import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.geometry import alignment_energy_function

def test_grid_energy_minima():
    """Test that energy is minimal at grid points."""
    pitch = 10.0
    # Create points exactly on grid
    coords = torch.tensor([[0.0, 0.0], [10.0, 20.0], [-10.0, 50.0]], dtype=torch.float32)
    
    energy = alignment_energy_function(coords, None, grid_pitch=pitch, lambda_grid=1.0, lambda_channel=0.0)
    
    assert torch.isclose(energy, torch.tensor(0.0), atol=1e-4)

def test_grid_energy_gradient():
    """Test that gradient points towards grid points."""
    pitch = 10.0
    # Point slightly off grid (11.0) -> should be pushed back to 10.0
    coords = torch.tensor([[11.0, 0.0]], dtype=torch.float32, requires_grad=True)
    
    energy = alignment_energy_function(coords, None, grid_pitch=pitch, lambda_grid=1.0, lambda_channel=0.0)
    energy.backward()
    
    grad = coords.grad
    # The gradient of sin^2(pi*x/K) at x=1.1K is positive (increasing energy), 
    # so we should move in negative direction to minimize.
    # sin(1.1 pi) is negative, but sin^2 is positive and increasing away from pi.
    # Wait, sin(1.1 pi) = sin(pi + 0.1pi) = -sin(0.1pi).
    # d/dx sin^2(u) = 2 sin(u) cos(u) du/dx.
    # u = pi * x / K.
    # at x = 1.1 K, u = 1.1 pi.
    # sin(1.1 pi) < 0. cos(1.1 pi) < 0.
    # product > 0.
    # So gradient is positive.
    # To minimize energy, we move against gradient -> negative direction -> towards 10.0. Correct.
    
    assert grad[0, 0] > 0
    assert grad[0, 1] == 0 # Y is already aligned

def test_channel_energy():
    """Test alignment between connected nodes."""
    coords = torch.tensor([[0.0, 0.0], [5.0, 5.0]], dtype=torch.float32, requires_grad=True)
    # Edge between them
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    
    energy = alignment_energy_function(coords, edge_index, grid_pitch=10.0, lambda_grid=0.0, lambda_channel=1.0)
    
    # Energy = ||p1 - p2||^2 = 5^2 + 5^2 = 50
    assert torch.isclose(energy, torch.tensor(50.0))
    
    energy.backward()
    grad = coords.grad
    
    # Gradient for p1 should pull it towards p2 (positive x, positive y)
    # d/dp1 (p1-p2)^2 = 2(p1-p2)
    # p1-p2 = (-5, -5) -> grad = (-10, -10)
    # To minimize, move against grad -> (+10, +10) -> towards p2.
    # Wait, d/dx (x-c)^2 = 2(x-c). At x=0, c=5: 2(-5) = -10.
    # Gradient is negative. We descend gradient -> move positive -> towards 5. Correct.
    
    assert grad[0, 0] < 0 
    assert grad[0, 1] < 0
    assert grad[1, 0] > 0 # p2 pulled towards p1
    assert grad[1, 1] > 0
