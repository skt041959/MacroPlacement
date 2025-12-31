import torch
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import RestorationDataset
from torch_geometric.data import HeteroData

def test_dataset_init():
    dataset = RestorationDataset(num_samples=10)
    assert len(dataset) == 10

def test_dataset_get_item():
    dataset = RestorationDataset(num_samples=5, count=20)
    data = dataset[0]
    
    assert isinstance(data, HeteroData)
    assert 'macro' in data.x_dict
    assert data['macro'].x.shape[0] == 20
    # Aligned coordinates as target
    assert hasattr(data, 'y_coords')
    assert data.y_coords.shape == (20, 2)
