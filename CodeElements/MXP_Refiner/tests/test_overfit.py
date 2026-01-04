import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    from src.config import Config
    from src.model import GraphUNet
    from src.dataset import RestorationDataset
    from src.diffusion import DiffusionScheduler
except ImportError:
    from config import Config
    from model import GraphUNet
    from dataset import RestorationDataset
    from diffusion import DiffusionScheduler

class SimpleMLP(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 + 1, hidden_dim), # +1 for t
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict, t):
        # Flatten input: x [N, 4], t [B]
        x = x_dict['macro']
        
        # Get batch vector
        num_nodes = x.size(0)
        batch = x_dict.get('batch', torch.zeros(num_nodes, dtype=torch.long, device=x.device))
        
        # Expand t to node level: [B] -> [N]
        t_expanded = t[batch]
        
        # Normalize time
        t_feat = t_expanded.float().view(-1, 1) / 100.0
        
        in_feat = torch.cat([x, t_feat], dim=-1)
        return self.net(in_feat)

def test_overfit_single_batch():
    """
    Sanity check: Can the model overfit a single batch of data?
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Overfitting test on {device}")
    
    # Use a small fixed dataset
    ds = RestorationDataset(num_samples=32, count=20, mode='grid', seed=42)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    batch = next(iter(loader)).to(device)
    
    # Use SimpleMLP to verify data/diffusion loop
    model = SimpleMLP(hidden_dim=256).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    diffusion = DiffusionScheduler(num_steps=100)
    diffusion.sqrt_alphas_cumprod = diffusion.sqrt_alphas_cumprod.to(device)
    diffusion.sqrt_one_minus_alphas_cumprod = diffusion.sqrt_one_minus_alphas_cumprod.to(device)
    
    losses = []
    
    # Train for 500 steps on the SAME batch
    for i in range(500):
        optimizer.zero_grad()
        
        # Fixed timestep for deterministic check or random?
        # Standard: Random t
        t = torch.randint(0, diffusion.num_steps, (batch.num_graphs,), device=device)
        t_expanded = t[batch['macro'].batch]
        
        noisy_x, noise = diffusion.add_noise(batch.y_coords, t_expanded)
        
        x_dict = batch.x_dict.copy()
        x_dict['macro'] = torch.cat([noisy_x, batch.x_dict['macro'][:, 2:]], dim=-1)
        
        edge_attr_dict = {
            ('macro', 'phys_edge', 'macro'): batch['macro', 'phys_edge', 'macro'].edge_attr,
            ('macro', 'logic_edge', 'macro'): batch['macro', 'logic_edge', 'macro'].edge_attr,
            ('macro', 'align_edge', 'macro'): None 
        }
        
        pred_noise = model(x_dict, batch.edge_index_dict, edge_attr_dict, t)
        loss = criterion(pred_noise, noise)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if i % 20 == 0:
            print(f"Step {i}: Loss {loss.item():.6f}")
            
    # Check if loss decreased significantly
    initial_loss = losses[0]
    final_loss = losses[-1]
    print(f"Initial: {initial_loss:.4f}, Final: {final_loss:.4f}")
    
    assert final_loss < 0.1 * initial_loss, "Model failed to overfit single batch! (Loss didn't drop by 90%)"
    print("SUCCESS: Model overfitted single batch.")

if __name__ == "__main__":
    test_overfit_single_batch()
