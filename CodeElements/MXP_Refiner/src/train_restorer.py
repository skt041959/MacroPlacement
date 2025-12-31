import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from config import Config
from model import FloorplanRestorer
from dataset import RestorationDataset

def train_restorer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 1. Dataset & Loader
    # Using small dataset for prototype
    dataset = RestorationDataset(num_samples=200, seed=Config.SEED)
    loader = DataLoader(dataset, batch_size=Config.RESTORER_BATCH_SIZE, shuffle=True)

    # 2. Model, Optimizer, Loss
    model = FloorplanRestorer(
        hidden_dim=Config.HIDDEN_DIM, 
        num_layers=Config.NUM_LAYERS, 
        num_heads=Config.NUM_HEADS
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.RESTORER_LR)
    criterion = nn.MSELoss()

    model.train()
    
    # History for tracking
    history = {'losses': []}

    for epoch in range(Config.RESTORER_EPOCHS):
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{Config.RESTORER_EPOCHS}")
        
        for data in pbar:
            data = data.to(device)
            
            # Construct edge_attr_dict
            edge_attr_dict = {
                ('macro', 'phys_edge', 'macro'): data['macro', 'phys_edge', 'macro'].edge_attr,
                ('macro', 'logic_edge', 'macro'): data['macro', 'logic_edge', 'macro'].edge_attr,
                ('macro', 'align_edge', 'macro'): None 
            }
            
            optimizer.zero_grad()
            
            # Forward
            pred_coords = model(data.x_dict, data.edge_index_dict, edge_attr_dict)
            
            # 1. Coordinate MSE Loss
            mse_loss = criterion(pred_coords, data.y_coords)
            
            # 2. Alignment Loss
            # align_edge_index: [2, Num_Align_Edges]
            align_edge_index = data.edge_index_dict[('macro', 'align_edge', 'macro')]
            if align_edge_index.shape[1] > 0:
                src_idx = align_edge_index[0]
                dst_idx = align_edge_index[1]
                
                # We want pred_coords[src] to be close to pred_coords[dst] in at least one dimension
                # But more simply, if they are connected by align_edge, they should follow the alignment of the target.
                # In our synthetic data, align_edge means they share X or Y.
                
                # Get ground truth alignment to know which axis to penalize
                # Or just penalize the difference if they are supposed to be aligned.
                # Let's check ground truth to see if it's X or Y alignment
                gt_src = data.y_coords[src_idx]
                gt_dst = data.y_coords[dst_idx]
                
                # mask_x: 1 if x aligned in GT, 0 otherwise
                mask_x = (torch.abs(gt_src[:, 0] - gt_dst[:, 0]) < 1e-5).float()
                mask_y = (torch.abs(gt_src[:, 1] - gt_dst[:, 1]) < 1e-5).float()
                
                diff = pred_coords[src_idx] - pred_coords[dst_idx]
                
                # Loss on X difference if X aligned, Loss on Y difference if Y aligned
                align_loss = (torch.pow(diff[:, 0], 2) * mask_x + torch.pow(diff[:, 1], 2) * mask_y).mean()
            else:
                align_loss = torch.tensor(0.0).to(device)
            
            loss = mse_loss + 0.1 * align_loss
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.6f}", 'MSE': f"{mse_loss.item():.6f}", 'Align': f"{align_loss.item():.6f}"})
            
        avg_loss = epoch_loss / len(loader)
        history['losses'].append(avg_loss)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.6f}")

    # 3. Save Model
    torch.save(model.state_dict(), "restorer_model.pth")
    print("Training finished. Model saved to restorer_model.pth")
    return history

if __name__ == "__main__":
    train_restorer()
