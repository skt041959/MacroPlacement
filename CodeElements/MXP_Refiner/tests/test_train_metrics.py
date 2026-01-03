import pytest
from unittest.mock import MagicMock, patch, mock_open
import torch
import sys
import os
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from train_restorer import train_restorer
from config import Config

@patch('train_restorer.DataLoader')
@patch('train_restorer.RestorationDataset')
@patch('train_restorer.GraphUNet')
@patch('train_restorer.optim')
@patch('train_restorer.plt')
@patch('train_restorer.compute_metrics')
@patch('torch.save')
@patch('csv.writer')
@patch('train_restorer.DiffusionScheduler')
def test_train_restorer_logging(mock_diff_cls, mock_csv_writer, mock_save, mock_compute_metrics, mock_plt, mock_optim, mock_model_cls, mock_dataset_cls, mock_loader):
    # Setup Config for fast test
    Config.RESTORER_EPOCHS = 1
    Config.RESTORER_BATCH_SIZE = 2
    Config.VAL_DATA_PATH = "dummy_val.pt"
    Config.TRAIN_DATA_PATH = "dummy_train.pt"
    
    # Identify device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Mock Scheduler
    mock_diff = MagicMock()
    mock_diff_cls.return_value = mock_diff
    mock_diff.num_steps = 100 
    
    # Mock tensors on the correct device
    mock_diff.sqrt_alphas_cumprod = torch.ones(101, device=device)
    mock_diff.sqrt_one_minus_alphas_cumprod = torch.zeros(101, device=device)
    
    def add_noise_side_effect(x_0, t, noise=None):
        if noise is None:
            noise = torch.zeros_like(x_0)
        return torch.zeros_like(x_0, requires_grad=True), noise
        
    mock_diff.add_noise.side_effect = add_noise_side_effect
    
    # Mock Model
    mock_model = MagicMock()
    mock_model_cls.return_value = mock_model
    mock_model.to.return_value = mock_model 
    
    def model_side_effect(x_dict, *args, **kwargs):
        return torch.zeros(x_dict['macro'].size(0), 2, requires_grad=True, device=x_dict['macro'].device)
        
    mock_model.side_effect = model_side_effect
    
    # Mock Optimizer
    mock_optimizer_instance = MagicMock()
    mock_optimizer_instance.param_groups = [{'lr': 0.001}]
    mock_optim.Adam.return_value = mock_optimizer_instance
    
    # Mock LR Scheduler
    mock_lr_scheduler = MagicMock()
    mock_optim.lr_scheduler.CosineAnnealingLR.return_value = mock_lr_scheduler
    
    # Mock Data
    class MockData:
        def __init__(self, dev='cpu'):
            self.dev = dev
            self.x_dict = {'macro': torch.zeros(2, 4, device=dev)}
            self.x_dict['macro'].batch = torch.tensor([0, 0], dtype=torch.long, device=dev)
            self.edge_index_dict = {
                ('macro', 'phys_edge', 'macro'): torch.zeros(2, 0, dtype=torch.long, device=dev),
                ('macro', 'logic_edge', 'macro'): torch.zeros(2, 0, dtype=torch.long, device=dev),
                ('macro', 'align_edge', 'macro'): torch.zeros(2, 0, dtype=torch.long, device=dev)
            }
            self.y_coords = torch.zeros(2, 2, device=dev)
            self.num_graphs = 1
            self.info_dict = {
                'aligned': [{'x':0,'y':0,'w':1,'h':1}, {'x':1,'y':1,'w':1,'h':1}],
                'disturbed': [{'x':0,'y':0,'w':1,'h':1}, {'x':1,'y':1,'w':1,'h':1}]
            }
            
        def to(self, target_dev): 
            return MockData(dev=target_dev)
            
        def __getitem__(self, key):
            if key == 'macro':
                m = MagicMock()
                m.batch = self.x_dict['macro'].batch
                m.edge_attr = None
                return m
            m = MagicMock()
            m.edge_attr = None
            return m

    mock_loader.return_value = [MockData(dev=device.type)]
    
    # Mock Metrics
    mock_compute_metrics.return_value = {
        'mse': 0.1,
        'overlap_aligned': 0,
        'overlap_disturbed': 100,
        'overlap_restored': 50,
        'alignment_recovery': 0.8
    }
    
    # Run training
    with patch("builtins.open", mock_open()) as mock_file:
        train_restorer()
        
        mock_writer_instance = mock_csv_writer.return_value
        
        # Header check
        header_call = mock_writer_instance.writerow.call_args_list[0]
        header = header_call[0][0]
        assert "Val_Overlap" in header
        assert "Val_Align" in header
        
        # Data row check
        data_call = mock_writer_instance.writerow.call_args_list[1]
        data = data_call[0][0]
        # [epoch, loss, val_mse, val_overlap, val_align, lr]
        assert len(data) == 6
        assert data[3] == 50.0 # Val_Overlap
        assert data[4] == 0.8 # Val_Align