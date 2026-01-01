import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import os

from config import Config
from model import FloorplanRestorer
from dataset import RestorationDataset

def train_restorer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 1. Dataset & Loader
    dataset = RestorationDataset(num_samples=Config.NUM_TRAIN_SAMPLES, seed=Config.SEED, path=Config.DATASET_PATH)
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
    history = {'loss': [], 'mse': [], 'align': []}
    
    # Open log file
    log_file = open("training.log", "w", newline='')
    writer = csv.writer(log_file)
    writer.writerow(["Epoch", "Loss", "MSE", "Align"])

    for epoch in range(Config.RESTORER_EPOCHS):
        epoch_loss = 0
        epoch_mse = 0
        epoch_align = 0
        
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
            align_edge_index = data.edge_index_dict[('macro', 'align_edge', 'macro')]
            if align_edge_index.shape[1] > 0:
                src_idx = align_edge_index[0]
                dst_idx = align_edge_index[1]
                
                gt_src = data.y_coords[src_idx]
                gt_dst = data.y_coords[dst_idx]
                
                mask_x = (torch.abs(gt_src[:, 0] - gt_dst[:, 0]) < 1e-5).float()
                mask_y = (torch.abs(gt_src[:, 1] - gt_dst[:, 1]) < 1e-5).float()
                
                diff = pred_coords[src_idx] - pred_coords[dst_idx]
                
                align_loss = (torch.pow(diff[:, 0], 2) * mask_x + torch.pow(diff[:, 1], 2) * mask_y).mean()
            else:
                align_loss = torch.tensor(0.0).to(device)
            
            loss = mse_loss + 0.1 * align_loss
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_align += align_loss.item()
            
            pbar.set_postfix({'Loss': f"{loss.item():.6f}", 'MSE': f"{mse_loss.item():.6f}"})
            
        avg_loss = epoch_loss / len(loader)
        avg_mse = epoch_mse / len(loader)
        avg_align = epoch_align / len(loader)
        
        history['loss'].append(avg_loss)
        history['mse'].append(avg_mse)
        history['align'].append(avg_align)
        
        writer.writerow([epoch+1, avg_loss, avg_mse, avg_align])
        log_file.flush()

    log_file.close()

    # 3. Save Model
    torch.save(model.state_dict(), "restorer_model.pth")
    print("Training finished. Model saved to restorer_model.pth")
    
    # 4. Plot Curves
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Total Loss')
    plt.plot(history['mse'], label='MSE Loss')
    plt.plot(history['align'], label='Align Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Restoration Model Training History')
    plt.grid(True)
    plt.savefig('training_curves.png')
    print("Training curves saved to training_curves.png")
    
    return history

if __name__ == "__main__":
    train_restorer()