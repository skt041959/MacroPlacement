import torch
import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Use simple imports if src is in path
try:
    from dataset import RestorationDataset
    from config import Config
except ImportError:
    from src.dataset import RestorationDataset
    from src.config import Config

def test_normalization_range():
    """Test that data is normalized to [-1, 1]."""
    # Create small dataset
    ds = RestorationDataset(num_samples=5, count=10, mode='random')
    data = ds[0]
    
    # Check node features: [x, y, w, h]
    # x, y should be [-1, 1]
    # w, h should be [0, 1]
    
    feats = data['macro'].x
    x, y = feats[:, 0], feats[:, 1]
    w, h = feats[:, 2], feats[:, 3]
    
    assert x.min() >= -1.0 and x.max() <= 1.0
    assert y.min() >= -1.0 and y.max() <= 1.0
    
    # Check targets
    targets = data.y_coords
    tx, ty = targets[:, 0], targets[:, 1]
    
    assert tx.min() >= -1.0 and tx.max() <= 1.0
    assert ty.min() >= -1.0 and ty.max() <= 1.0

def test_augmentation_diversity():
    """Test that augmentation produces different outputs for same index."""
    ds = RestorationDataset(num_samples=1, count=5, mode='random')
    
    # Probability of NO change is (0.5)^3 = 0.125.
    # Probability of change is 0.875.
    # If we sample 10 times, highly likely to see difference.
    
    first = ds[0]['macro'].x
    seen_diff = False
    
    for _ in range(20):
        current = ds[0]['macro'].x
        if not torch.allclose(first, current):
            seen_diff = True
            break
            
    assert seen_diff, "Augmentation should produce diverse samples."
