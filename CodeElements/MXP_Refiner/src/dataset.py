import torch
from torch.utils.data import Dataset
from generator import SyntheticDataGenerator
from data_builder import GraphBuilder
from config import Config
import numpy as np

class RestorationDataset(Dataset):
    def __init__(self, num_samples=100, count=None, mode=None, noise_level=None, seed=None):
        self.num_samples = num_samples
        self.count = count if count is not None else Config.MACRO_COUNT
        self.mode = mode if mode is not None else Config.GENERATION_MODE
        self.noise_level = noise_level if noise_level is not None else Config.NOISE_LEVEL
        
        self.generator = SyntheticDataGenerator(seed=seed, 
                                               canvas_width=Config.CANVAS_WIDTH, 
                                               canvas_height=Config.CANVAS_HEIGHT)
        
        # We pre-generate or generate on the fly? 
        # On the fly is better for diversity if we don't reset seed every time.
        # But for fixed dataset size, we might want reproducibility.
        # Let's generate a list of (aligned, disturbed) macro sets.
        self.samples = []
        for _ in range(num_samples):
            # Pass grid_cols if needed, but generator defaults or uses kwargs
            aligned, disturbed = self.generator.generate(count=self.count, 
                                                       mode=self.mode, 
                                                       noise_level=self.noise_level,
                                                       grid_cols=Config.GRID_COLS)
            self.samples.append((aligned, disturbed))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        aligned, disturbed = self.samples[idx]
        
        # Build graph from disturbed layout (the input to the model)
        # Dummy netlist for now as per previous phase logic
        builder = GraphBuilder(disturbed, netlist=[])
        data = builder.build_hetero_graph()
        
        # Target: Aligned coordinates
        # Normalize target coordinates same as node features
        target_coords = []
        for m in aligned:
            # We want (x, y) centers normalized
            cx = m['x'] + m['w'] / 2
            cy = m['y'] + m['h'] / 2
            target_coords.append([
                cx / Config.CANVAS_WIDTH,
                cy / Config.CANVAS_HEIGHT
            ])
            
        data.y_coords = torch.tensor(target_coords, dtype=torch.float)
        
        return data
