import torch
from torch.utils.data import Dataset
from generator import SyntheticDataGenerator
from data_builder import GraphBuilder
from config import Config
import numpy as np
import os
from tqdm import tqdm

class RestorationDataset(Dataset):
    def __init__(self, num_samples=100, count=None, mode=None, noise_level=None, seed=None, path=None):
        self.num_samples = num_samples
        self.count = count if count is not None else Config.MACRO_COUNT
        self.mode = mode if mode is not None else Config.GENERATION_MODE
        self.noise_level = noise_level if noise_level is not None else Config.NOISE_LEVEL
        
        if path and os.path.exists(path):
            print(f"Loading dataset from {path}")
            self.data_list = torch.load(path, weights_only=False)
            self.num_samples = len(self.data_list)
        else:
            self.generator = SyntheticDataGenerator(seed=seed, 
                                                   canvas_width=Config.CANVAS_WIDTH, 
                                                   canvas_height=Config.CANVAS_HEIGHT)
            
            self.data_list = []
            print(f"Generating {num_samples} samples...")
            for i in tqdm(range(num_samples), desc="Generating Data"):
                aligned, disturbed = self.generator.generate(count=self.count, 
                                                           mode=self.mode, 
                                                           noise_level=self.noise_level,
                                                           grid_cols=Config.GRID_COLS)
                
                builder = GraphBuilder(disturbed, netlist=[])
                data = builder.build_hetero_graph()
                
                target_coords = []
                for m in aligned:
                    cx = m['x'] + m['w'] / 2
                    cy = m['y'] + m['h'] / 2
                    target_coords.append([
                        cx / Config.CANVAS_WIDTH,
                        cy / Config.CANVAS_HEIGHT
                    ])
                    
                data.y_coords = torch.tensor(target_coords, dtype=torch.float)
                
                # Store original macro dicts in info_dict for visualization
                data.info_dict = {
                    'aligned': aligned,
                    'disturbed': disturbed,
                    'category': self.mode
                }
                
                self.data_list.append(data)
            
            if path:
                print(f"Saving dataset to {path}")
                torch.save(self.data_list, path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
