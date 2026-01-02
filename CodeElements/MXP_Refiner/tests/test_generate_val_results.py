import pytest
from unittest.mock import MagicMock, patch
import torch
import sys
import os

# Sys path hack to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

@patch('torch.load')
@patch('torch.save')
@patch('model.GraphToSeqRestorer')
@patch('evaluate_model.compute_metrics')
def test_generate_val_results(mock_metrics, mock_model_cls, mock_save, mock_load):
    from generate_val_results import generate_results
    
    # Setup mocks
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model # Fix: .to() should return self
    mock_model_cls.return_value = mock_model
    mock_model.eval.return_value = None
    # Mock inference output: tensor of shape (2, 2)
    mock_model.return_value = torch.tensor([[0.5, 0.5], [0.6, 0.6]])
    
    # Mock Dataset content
    mock_data = MagicMock()
    # Mock info_dict
    mock_data.info_dict = {
        'aligned': [{'x': 0, 'y': 0, 'w': 10, 'h': 10}],
        'disturbed': [{'x': 1, 'y': 1, 'w': 10, 'h': 10}]
    }
    # Mock GNN input attributes
    mock_data.x_dict = {}
    mock_data.edge_index_dict = {('macro', 'align_edge', 'macro'): torch.tensor([])}
    # edge_attr for phys/logic need to be present or handled
    # The actual code usually accesses data['macro', 'phys_edge', 'macro'].edge_attr
    # So we need to mock __getitem__ or attributes deeply?
    # Let's mock the attribute access
    mock_edge_store = MagicMock()
    mock_edge_store.edge_attr = torch.tensor([])
    mock_data.__getitem__.return_value = mock_edge_store 
    
    # But wait, data['macro', 'phys_edge', 'macro'] uses __getitem__ on HeteroData
    # Let's simplify and assume the script will handle extracting edge_attr safely or we mock the object properly
    # Using a simpler approach:
    class MockHeteroData:
        def __init__(self):
            self.info_dict = {
                'aligned': [{'x': 0.0, 'y': 0.0, 'w': 10.0, 'h': 10.0}],
                'disturbed': [{'x': 1.0, 'y': 1.0, 'w': 10.0, 'h': 10.0}]
            }
            self.x_dict = {}
            self.edge_index_dict = {('macro', 'align_edge', 'macro'): torch.zeros(2, 0)}
            
        def __getitem__(self, key):
            # Mock edge stores
            mock_store = MagicMock()
            mock_store.edge_attr = torch.tensor([])
            return mock_store
            
        def to(self, device):
            return self

    val_data_list = [MockHeteroData()]
    
    def load_side_effect(path, **kwargs):
        if "val_dataset" in path:
            return val_data_list
        else:
            return {} # Mock weights

    mock_load.side_effect = load_side_effect
    
    mock_metrics.return_value = {'mse': 0.1, 'overlap': 0.0}

    # Run the function
    generate_results()
    
    # Assertions
    mock_load.assert_called() # Should load model and dataset
    mock_model.load_state_dict.assert_called()
    mock_save.assert_called() # Should save results
    
    # Verify structure of saved data
    saved_data = mock_save.call_args[0][0]
    assert len(saved_data) == 1
    assert 'aligned' in saved_data[0]
    assert 'disturbed' in saved_data[0]
    assert 'restored' in saved_data[0]
    assert 'metrics' in saved_data[0]
