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
@patch('train_restorer.GraphToSeqRestorer')
@patch('train_restorer.optim')
@patch('train_restorer.plt')
@patch('train_restorer.compute_metrics')
@patch('torch.save')
@patch('csv.writer')
def test_train_restorer_logging(mock_csv_writer, mock_save, mock_compute_metrics, mock_plt, mock_optim, mock_model_cls, mock_dataset_cls, mock_loader):
    # Setup Config for fast test
    Config.RESTORER_EPOCHS = 1
    Config.RESTORER_BATCH_SIZE = 2
    Config.VAL_DATA_PATH = "dummy_val.pt"
    Config.TRAIN_DATA_PATH = "dummy_train.pt"
    
    # Mock Model
    mock_model = MagicMock()
    mock_model_cls.return_value = mock_model
    mock_model.to.return_value = mock_model # Fix .to()
    mock_model.return_value = torch.zeros(2, 2, requires_grad=True) # 2 nodes, 2 coords
    
    # Mock Optimizer
    mock_optimizer_instance = MagicMock()
    mock_optimizer_instance.param_groups = [{'lr': 0.001}]
    mock_optim.Adam.return_value = mock_optimizer_instance
    
    # Mock Scheduler
    mock_scheduler = MagicMock()
    mock_optim.lr_scheduler.CosineAnnealingLR.return_value = mock_scheduler
    
    # Mock Data
    class MockData:
        def __init__(self):
            self.x_dict = {'macro': torch.zeros(2, 4)}
            # Add batch vector
            self.x_dict['macro'].batch = torch.tensor([0, 0], dtype=torch.long) # 1 graph, 2 nodes
            self.edge_index_dict = {
                ('macro', 'phys_edge', 'macro'): torch.zeros(2, 0, dtype=torch.long),
                ('macro', 'logic_edge', 'macro'): torch.zeros(2, 0, dtype=torch.long),
                ('macro', 'align_edge', 'macro'): torch.zeros(2, 0, dtype=torch.long)
            }
            self.y_coords = torch.zeros(2, 2)
            self.num_graphs = 1
            self.info_dict = {
                'aligned': [{'x':0,'y':0,'w':1,'h':1}, {'x':1,'y':1,'w':1,'h':1}],
                'disturbed': [{'x':0,'y':0,'w':1,'h':1}, {'x':1,'y':1,'w':1,'h':1}]
            }
            
        def to(self, device): return self
        def __getitem__(self, key):
            if key == 'macro':
                # Return an object that has .batch
                m = MagicMock()
                m.batch = self.x_dict['macro'].batch
                return m
            m = MagicMock()
            m.edge_attr = None
            return m

    mock_loader.return_value = [MockData()]
    
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
        
        # Verify calls to csv.writer
        mock_writer_instance = mock_csv_writer.return_value
        
        # Header
        header_call = mock_writer_instance.writerow.call_args_list[0]
        header = header_call[0][0]
        assert "Val_Overlap" in header
        assert "Val_Align" in header
        
        # Data
        data_call = mock_writer_instance.writerow.call_args_list[1]
        data = data_call[0][0]
        # Epoch, Loss, MSE, Align, Val_MSE, Val_Overlap, Val_Align, LR
        assert len(data) == 8
        assert data[5] == 50.0 # Val_Overlap
        assert data[6] == 0.8 # Val_Align
