import torch
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import FloorplanRestorer
from torch_geometric.data import HeteroData

def test_restorer_init():
    model = FloorplanRestorer(hidden_dim=64, num_layers=2, num_heads=4)
    assert isinstance(model, torch.nn.Module)

def test_restorer_forward():
    hidden_dim = 64
    model = FloorplanRestorer(hidden_dim=hidden_dim, num_layers=2, num_heads=4)
    
    # Mock HeteroData
    data = HeteroData()
    num_nodes = 10
    data['macro'].x = torch.randn(num_nodes, 4)
    
    # Simple edges
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data['macro', 'phys_edge', 'macro'].edge_index = edge_index
    data['macro', 'phys_edge', 'macro'].edge_attr = torch.randn(2, 1)
    
    data['macro', 'align_edge', 'macro'].edge_index = edge_index
    
    data['macro', 'logic_edge', 'macro'].edge_index = edge_index
    data['macro', 'logic_edge', 'macro'].edge_attr = torch.randn(2, 1)
    
    edge_attr_dict = {
        ('macro', 'phys_edge', 'macro'): data['macro', 'phys_edge', 'macro'].edge_attr,
        ('macro', 'logic_edge', 'macro'): data['macro', 'logic_edge', 'macro'].edge_attr,
        ('macro', 'align_edge', 'macro'): None
    }
    
    # Forward pass
    # The decoder should output restored coordinates [N, 2] (x, y)
    out = model(data.x_dict, data.edge_index_dict, edge_attr_dict)
    
    assert out.shape == (num_nodes, 2)

def test_refiner_refactored():
    from src.model import HeteroGATRefiner
    model = HeteroGATRefiner(hidden_dim=64, out_dim=7, num_layers=2, num_heads=4)
    
    # Mock HeteroData
    data = HeteroData()
    num_nodes = 10
    data['macro'].x = torch.randn(num_nodes, 4)
    
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data['macro', 'phys_edge', 'macro'].edge_index = edge_index
    data['macro', 'phys_edge', 'macro'].edge_attr = torch.randn(2, 1)
    data['macro', 'align_edge', 'macro'].edge_index = edge_index
    data['macro', 'logic_edge', 'macro'].edge_index = edge_index
    data['macro', 'logic_edge', 'macro'].edge_attr = torch.randn(2, 1)
    
    edge_attr_dict = {
        ('macro', 'phys_edge', 'macro'): data['macro', 'phys_edge', 'macro'].edge_attr,
        ('macro', 'logic_edge', 'macro'): data['macro', 'logic_edge', 'macro'].edge_attr,
        ('macro', 'align_edge', 'macro'): None
    }
    
    action_logits, value = model(data.x_dict, data.edge_index_dict, edge_attr_dict)
    
    assert action_logits.shape == (num_nodes, 7)
    assert value.shape == (1,)
