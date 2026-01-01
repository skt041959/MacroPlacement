import torch
import numpy as np
from config import Config
from model import GraphToSeqRestorer
from data_builder import GraphBuilder

class FloorplanRestorationInference:
    def __init__(self, model_path="restorer_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GraphToSeqRestorer(
            hidden_dim=Config.HIDDEN_DIM, 
            num_layers=Config.NUM_LAYERS, 
            num_heads=Config.NUM_HEADS
        ).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            print(f"Loaded model from {model_path}")
        except FileNotFoundError:
            print(f"Warning: {model_path} not found. Using uninitialized model.")
            
        self.model.eval()

    def restore(self, disturbed_macros):
        """
        disturbed_macros: list of dicts {'id', 'x', 'y', 'w', 'h'}
        Returns: restored_macros (same format)
        """
        # 1. Build graph from disturbed
        builder = GraphBuilder(disturbed_macros, netlist=[])
        data = builder.build_hetero_graph().to(self.device)
        
        # 2. Construct edge_attr_dict
        edge_attr_dict = {
            ('macro', 'phys_edge', 'macro'): data['macro', 'phys_edge', 'macro'].edge_attr,
            ('macro', 'logic_edge', 'macro'): data['macro', 'logic_edge', 'macro'].edge_attr,
            ('macro', 'align_edge', 'macro'): None 
        }
        
        # 3. Forward
        with torch.no_grad():
            pred_coords = self.model(data.x_dict, data.edge_index_dict, edge_attr_dict)
            
        # 4. Denormalize coordinates
        pred_coords = pred_coords.cpu().numpy() # [N, 2]
        
        restored_macros = []
        for i, m in enumerate(disturbed_macros):
            new_m = m.copy()
            # pred_coords are centers normalized
            cx = pred_coords[i, 0] * Config.CANVAS_WIDTH
            cy = pred_coords[i, 1] * Config.CANVAS_HEIGHT
            
            # Convert back to bottom-left (x, y)
            new_m['x'] = float(cx - m['w'] / 2)
            new_m['y'] = float(cy - m['h'] / 2)
            
            # Clip to canvas
            new_m['x'] = np.clip(new_m['x'], 0, Config.CANVAS_WIDTH - m['w'])
            new_m['y'] = np.clip(new_m['y'], 0, Config.CANVAS_HEIGHT - m['h'])
            
            restored_macros.append(new_m)
            
        return restored_macros

if __name__ == "__main__":
    from generator import SyntheticDataGenerator
    
    # Simple test
    gen = SyntheticDataGenerator(seed=42)
    aligned, disturbed = gen.generate(count=20, mode='clustered', noise_level=Config.NOISE_LEVEL)
    
    inference = FloorplanRestorationInference()
    restored = inference.restore(disturbed)
    
    print(f"Restored {len(restored)} macros.")
    print(f"First macro Aligned: ({aligned[0]['x']:.2f}, {aligned[0]['y']:.2f})")
    print(f"First macro Disturbed: ({disturbed[0]['x']:.2f}, {disturbed[0]['y']:.2f})")
    print(f"First macro Restored: ({restored[0]['x']:.2f}, {restored[0]['y']:.2f})")
