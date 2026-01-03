import torch
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import GraphUNet

def test_graph_unet_instantiation():
    hidden_dim = 64
    num_layers = 2
    num_heads = 4
    model = GraphUNet(hidden_dim, num_layers, num_heads)
    assert isinstance(model, torch.nn.Module)

def test_graph_unet_forward():
    hidden_dim = 64
    num_layers = 2
    num_heads = 4
    num_nodes = 10
    
    model = GraphUNet(hidden_dim, num_layers, num_heads)
    
    # x_dict: {'macro': [N, 4]}
    x_dict = {'macro': torch.randn(num_nodes, 4)}
    
    # edge_index_dict: dummy edges
    edge_index = torch.randint(0, num_nodes, (2, 20))
    edge_index_dict = {
        ('macro', 'phys_edge', 'macro'): edge_index,
        ('macro', 'align_edge', 'macro'): edge_index,
        ('macro', 'logic_edge', 'macro'): edge_index
    }
    
    # edge_attr_dict
    edge_attr = torch.randn(20, 1)
    edge_attr_dict = {
        ('macro', 'phys_edge', 'macro'): edge_attr,
        ('macro', 'align_edge', 'macro'): None,
        ('macro', 'logic_edge', 'macro'): edge_attr
    }
    
    out = model(x_dict, edge_index_dict, edge_attr_dict)
    
    # Output should be (x, y) for each node
    assert out.shape == (num_nodes, 2)

def test_graph_unet_variable_nodes():
    hidden_dim = 32
    num_layers = 1
    num_heads = 2
    model = GraphUNet(hidden_dim, num_layers, num_heads)
    
    for n in [5, 20, 50]:
        x_dict = {'macro': torch.randn(n, 4)}
        edge_index = torch.randint(0, n, (2, n * 2))
        edge_index_dict = {
            ('macro', 'phys_edge', 'macro'): edge_index,
            ('macro', 'align_edge', 'macro'): edge_index,
            ('macro', 'logic_edge', 'macro'): edge_index
        }
        edge_attr = torch.randn(n * 2, 1)
        edge_attr_dict = {
            ('macro', 'phys_edge', 'macro'): edge_attr,
            ('macro', 'align_edge', 'macro'): None,
            ('macro', 'logic_edge', 'macro'): edge_attr
        }
        out = model(x_dict, edge_index_dict, edge_attr_dict)
        assert out.shape == (n, 2)
