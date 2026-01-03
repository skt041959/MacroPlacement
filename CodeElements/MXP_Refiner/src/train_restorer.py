import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import os
import numpy as np

from config import Config
from model import GraphUNet
from dataset import RestorationDataset
from evaluate_model import compute_metrics
from diffusion import DiffusionScheduler

def train_restorer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 1. Dataset & Loader
    train_dataset = RestorationDataset(path=Config.TRAIN_DATA_PATH)
    val_dataset = RestorationDataset(path=Config.VAL_DATA_PATH)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.RESTORER_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.RESTORER_BATCH_SIZE, shuffle=False)

    # 2. Model, Optimizer, Loss, Scheduler
    model = GraphUNet(
        hidden_dim=Config.HIDDEN_DIM, 
        num_layers=Config.NUM_LAYERS, 
        num_heads=Config.NUM_HEADS
    ).to(device)
    
    diffusion = DiffusionScheduler()
    # Move scheduler tensors to device if possible
    diffusion.sqrt_alphas_cumprod = diffusion.sqrt_alphas_cumprod.to(device)
    diffusion.sqrt_one_minus_alphas_cumprod = diffusion.sqrt_one_minus_alphas_cumprod.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.RESTORER_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.RESTORER_EPOCHS)
    criterion = nn.MSELoss()

    # History for tracking
    history = {'loss': [], 'val_mse': [], 'val_overlap': [], 'val_align': [], 'lr': []}
    
    # Open log file
    log_file = open("training.log", "w", newline='')
    writer = csv.writer(log_file)
    # Loss is noise prediction MSE
    writer.writerow(["Epoch", "Loss", "Val_MSE", "Val_Overlap", "Val_Align", "LR"])

    for epoch in range(Config.RESTORER_EPOCHS):
        model.train()
        epoch_loss = 0
        current_lr = optimizer.param_groups[0]['lr']
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.RESTORER_EPOCHS} (LR: {current_lr:.6f})")
        
        for data in pbar:
            data = data.to(device)
            
            # 1. Sample random timesteps
            t = torch.randint(0, diffusion.num_steps, (data.num_graphs,), device=device)
            
            # 2. Add noise to coordinates (data.y_coords are ground truth x0)
            noisy_x, noise = diffusion.add_noise(data.y_coords, t[data['macro'].batch])
            
            # 3. Update x_dict with noisy coordinates
            # Current x_dict['macro'] has [cx, cy, w, h] where cx, cy is disturbed input.
            # We replace them with noisy_x
            x_dict = data.x_dict.copy()
            x_dict['macro'] = torch.cat([noisy_x, data.x_dict['macro'][:, 2:]], dim=-1)
            
            # 4. Forward: Predict noise
            edge_attr_dict = {
                ('macro', 'phys_edge', 'macro'): data['macro', 'phys_edge', 'macro'].edge_attr,
                ('macro', 'logic_edge', 'macro'): data['macro', 'logic_edge', 'macro'].edge_attr,
                ('macro', 'align_edge', 'macro'): None 
            }
            
            optimizer.zero_grad()
            pred_noise = model(x_dict, data.edge_index_dict, edge_attr_dict)
            
            # 5. Loss: MSE between pred and truth noise
            loss = criterion(pred_noise, noise)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.6f}"})
            
        # Validation
        model.eval()
        val_mse = 0
        val_overlap = 0
        val_align_recovery = 0
        total_graphs = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                
                # Simple validation: run model once at mid-timestep or just MSE on noise
                # For better tracking, we'll do mid-point noise prediction
                t = torch.full((data.num_graphs,), diffusion.num_steps // 2, device=device, dtype=torch.long)
                noisy_x, noise = diffusion.add_noise(data.y_coords, t[data['macro'].batch])
                x_dict = data.x_dict.copy()
                x_dict['macro'] = torch.cat([noisy_x, data.x_dict['macro'][:, 2:]], dim=-1)
                
                edge_attr_dict = {
                    ('macro', 'phys_edge', 'macro'): data['macro', 'phys_edge', 'macro'].edge_attr,
                    ('macro', 'logic_edge', 'macro'): data['macro', 'logic_edge', 'macro'].edge_attr,
                    ('macro', 'align_edge', 'macro'): None 
                }
                pred_noise = model(x_dict, data.edge_index_dict, edge_attr_dict)
                val_mse += criterion(pred_noise, noise).item() * data.num_graphs
                
                # Approximate restoration for overlap/align metrics (Mid-point estimate)
                # x0_pred = (xt - sqrt(1-alpha)*eps_pred) / sqrt(alpha)
                sqrt_alpha = diffusion.sqrt_alphas_cumprod[t].view(-1, 1)[data['macro'].batch]
                sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)[data['macro'].batch]
                pred_x0 = (noisy_x - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha
                
                # Unbatch for metrics
                batch_vector = data['macro'].batch
                for g_idx in range(data.num_graphs):
                    mask = (batch_vector == g_idx)
                    feats = data.x_dict['macro'][mask]
                    pred = pred_x0[mask]
                    target = data.y_coords[mask]
                    
                    w_raw = feats[:, 2] * Config.CANVAS_WIDTH
                    h_raw = feats[:, 3] * Config.CANVAS_HEIGHT
                    
                    # Aligned
                    aligned_cx = target[:, 0] * Config.CANVAS_WIDTH
                    aligned_cy = target[:, 1] * Config.CANVAS_HEIGHT
                    aligned_macros = []
                    for k in range(len(w_raw)):
                        aligned_macros.append({
                            'x': float(aligned_cx[k] - w_raw[k]/2),
                            'y': float(aligned_cy[k] - h_raw[k]/2),
                            'w': float(w_raw[k]),
                            'h': float(h_raw[k])
                        })
                    
                    # Disturbed
                    dist_cx = feats[:, 0] * Config.CANVAS_WIDTH
                    dist_cy = feats[:, 1] * Config.CANVAS_HEIGHT
                    disturbed_macros = []
                    for k in range(len(w_raw)):
                        disturbed_macros.append({
                            'x': float(dist_cx[k] - w_raw[k]/2),
                            'y': float(dist_cy[k] - h_raw[k]/2),
                            'w': float(w_raw[k]),
                            'h': float(h_raw[k])
                        })
                        
                    # Restored (Estimate)
                    rest_cx = pred[:, 0] * Config.CANVAS_WIDTH
                    rest_cy = pred[:, 1] * Config.CANVAS_HEIGHT
                    restored_macros = []
                    for k in range(len(w_raw)):
                        restored_macros.append({
                            'x': float(rest_cx[k] - w_raw[k]/2),
                            'y': float(rest_cy[k] - h_raw[k]/2),
                            'w': float(w_raw[k]),
                            'h': float(h_raw[k])
                        })
                        
                    m = compute_metrics(aligned_macros, disturbed_macros, restored_macros)
                    val_overlap += m['overlap_restored']
                    val_align_recovery += m['alignment_recovery']
                    
                total_graphs += data.num_graphs
        
        avg_val_mse = val_mse / total_graphs
        avg_val_overlap = val_overlap / total_graphs
        avg_val_align = val_align_recovery / total_graphs
        
        # Step the scheduler
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        
        history['loss'].append(avg_loss)
        history['val_mse'].append(avg_val_mse)
        history['val_overlap'].append(avg_val_overlap)
        history['val_align'].append(avg_val_align)
        history['lr'].append(current_lr)
        
        print(f"Epoch {epoch+1} - Noise Loss: {avg_loss:.4f} | Val Noise MSE: {avg_val_mse:.4f} | Val Ov: {avg_val_overlap:.1f} | Val Align: {avg_val_align:.4f}")
        
        writer.writerow([epoch+1, avg_loss, avg_val_mse, avg_val_overlap, avg_val_align, current_lr])
        log_file.flush()

    log_file.close()

    # 3. Save Model
    torch.save(model.state_dict(), "restorer_model.pth")
    print("Training finished. Model saved to restorer_model.pth")
    
    # 4. Plot Curves
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Noise Loss')
    plt.plot(history['val_mse'], label='Val Noise MSE')
    plt.title('Denoising Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['val_overlap'], label='Val Overlap')
    plt.title('Overlap')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['val_align'], label='Val Align Recovery')
    plt.title('Alignment')
    plt.legend()
    
    plt.savefig('training_curves.png')
    print("Training curves saved to training_curves.png")
    
    return history

if __name__ == "__main__":
    train_restorer()