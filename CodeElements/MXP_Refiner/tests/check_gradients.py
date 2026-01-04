import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.model import GraphUNet
from src.dataset import RestorationDataset
from src.diffusion import DiffusionScheduler

def check_gradients():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Gradient check on {device}")
    
    # 1 batch
    ds = RestorationDataset(num_samples=4, count=20, mode='grid', seed=42)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader)).to(device)
    
    model = GraphUNet(hidden_dim=128, num_layers=4, num_heads=4).to(device)
    diffusion = DiffusionScheduler(num_steps=100)
    diffusion.sqrt_alphas_cumprod = diffusion.sqrt_alphas_cumprod.to(device)
    diffusion.sqrt_one_minus_alphas_cumprod = diffusion.sqrt_one_minus_alphas_cumprod.to(device)
    
    # Forward
    t = torch.randint(0, 100, (4,), device=device)
    t_expanded = t[batch['macro'].batch]
    
    noisy_x, noise = diffusion.add_noise(batch.y_coords, t_expanded)
    x_dict = batch.x_dict.copy()
    x_dict['macro'] = torch.cat([noisy_x, batch.x_dict['macro'][:, 2:]], dim=-1)
    
    for k, v in batch.edge_index_dict.items():
        batch.edge_index_dict[k] = v.to(device)
        
    edge_attr_dict = {
        ('macro', 'phys_edge', 'macro'): batch['macro', 'phys_edge', 'macro'].edge_attr,
        ('macro', 'logic_edge', 'macro'): batch['macro', 'logic_edge', 'macro'].edge_attr,
        ('macro', 'align_edge', 'macro'): None 
    }
    
    pred_noise = model(x_dict, batch.edge_index_dict, edge_attr_dict, t)
    loss = nn.MSELoss()(pred_noise, noise)
    
    print(f"Initial Loss: {loss.item()}")
    
    # Backward
    loss.backward()
    
    # Check Gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_mean = param.grad.abs().mean().item()
            grad_max = param.grad.abs().max().item()
            if grad_max == 0.0:
                 print(f"Warning: Zero gradient for {name}")
            # print(f"{name}: Mean={grad_mean:.6f}, Max={grad_max:.6f}")
        else:
            print(f"No gradient for {name}")
            
    if not has_grad:
        print("FAIL: No gradients computed!")
    else:
        print("Success: Gradients computed.")

if __name__ == "__main__":
    check_gradients()
