import unittest
from unittest.mock import patch, MagicMock
import torch
import os
import shutil
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.dataset import RestorationDataset
from src.train_restorer import train_restorer

class TestConvergence(unittest.TestCase):
    def setUp(self):
        self.temp_dir = "tests/temp_data"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.train_path = os.path.join(self.temp_dir, "train.pt")
        self.val_path = os.path.join(self.temp_dir, "val.pt")
        
        # Generate small datasets
        print("Generating temp datasets...")
        RestorationDataset(num_samples=32, count=10, mode='grid', seed=42, path=self.train_path)
        RestorationDataset(num_samples=8, count=10, mode='grid', seed=99, path=self.val_path)

    def tearDown(self):
        # Cleanup
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        if os.path.exists("restorer_model.pth"):
            # Don't delete user's model if it exists in root?
            # train_restorer writes to CWD.
            # We should run this test in a temp CWD or rename output.
            pass

    @patch('src.train_restorer.Config')
    def test_training_converges(self, MockConfig):
        # Configure Mock
        MockConfig.RESTORER_EPOCHS = 50
        MockConfig.RESTORER_BATCH_SIZE = 4
        MockConfig.RESTORER_LR = 1e-3
        MockConfig.HIDDEN_DIM = 64
        MockConfig.NUM_LAYERS = 2
        MockConfig.NUM_HEADS = 2
        MockConfig.TRAIN_DATA_PATH = self.train_path
        MockConfig.VAL_DATA_PATH = self.val_path
        MockConfig.CANVAS_WIDTH = 1000.0
        MockConfig.CANVAS_HEIGHT = 1000.0
        
        # Use only 4 samples for training to ensure overfitting
        RestorationDataset(num_samples=4, count=10, mode='grid', seed=42, path=self.train_path)
        
        # Need other configs used by train_restorer
        MockConfig.DIFFUSION_TIMESTEPS = 50
        MockConfig.BETA_START = 1e-4
        MockConfig.BETA_END = 0.02
        MockConfig.PHYS_EDGE_CUTOFF = 1500.0
        
        print("Starting training...")
        history = train_restorer()
        
        losses = history['loss']
        print(f"Losses: {losses}")
        
        self.assertTrue(losses[-1] < 0.3, f"Final loss too high: {losses[-1]}")

if __name__ == '__main__':
    unittest.main()
