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
from model import GraphToSeqRestorer
from dataset import RestorationDataset
from evaluate_model import compute_metrics

def train_restorer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 1. Dataset & Loader
    train_dataset = RestorationDataset(path=Config.TRAIN_DATA_PATH)
    val_dataset = RestorationDataset(path=Config.VAL_DATA_PATH)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.RESTORER_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.RESTORER_BATCH_SIZE, shuffle=False)

    # 2. Model, Optimizer, Loss
    model = GraphToSeqRestorer(
        hidden_dim=Config.HIDDEN_DIM, 
        num_layers=Config.NUM_LAYERS, 
        num_heads=Config.NUM_HEADS
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.RESTORER_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.RESTORER_EPOCHS)
    criterion = nn.MSELoss()

    # History for tracking
    history = {'loss': [], 'mse': [], 'align': [], 'val_mse': [], 'val_overlap': [], 'val_align': [], 'lr': []}
    
    # Open log file
    log_file = open("training.log", "w", newline='')
    writer = csv.writer(log_file)
    writer.writerow(["Epoch", "Loss", "MSE", "Align", "Val_MSE", "Val_Overlap", "Val_Align", "LR"])

    for epoch in range(Config.RESTORER_EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_mse = 0
        epoch_align = 0
        current_lr = optimizer.param_groups[0]['lr']
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.RESTORER_EPOCHS} (LR: {current_lr:.6f})")
        
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
            
        # Validation
        model.eval()
        val_mse = 0
        val_overlap = 0
        val_align_recovery = 0
        total_graphs = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                edge_attr_dict = {
                    ('macro', 'phys_edge', 'macro'): data['macro', 'phys_edge', 'macro'].edge_attr,
                    ('macro', 'logic_edge', 'macro'): data['macro', 'logic_edge', 'macro'].edge_attr,
                    ('macro', 'align_edge', 'macro'): None 
                }
                pred_coords = model(data.x_dict, data.edge_index_dict, edge_attr_dict)
                val_mse += criterion(pred_coords, data.y_coords).item() * data.num_graphs # Weighted avg if needed, but mean is fine if batch size const
                
                # Unbatch for metrics
                batch_vector = data['macro'].batch
                for g_idx in range(data.num_graphs):
                    mask = (batch_vector == g_idx)
                    feats = data.x_dict['macro'][mask]
                    pred = pred_coords[mask]
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
                        
                    # Restored
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
        
        # Calculate Averages
        # criterion is reduction='mean' (default), so val_mse accumulation above needs care.
        # DataLoader iterates over batches. criterion(pred, y) gives mean MSE for that batch.
        # Summing means and dividing by num_batches gives average mean MSE.
        # But val_overlap is summed per graph.
        
        avg_val_mse = val_mse / len(val_loader) # This assumes equal batch sizes roughly, acceptable
        avg_val_overlap = val_overlap / total_graphs
        avg_val_align = val_align_recovery / total_graphs
        
        # Step the scheduler
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        avg_mse = epoch_mse / len(train_loader)
        avg_align = epoch_align / len(train_loader)
        
        history['loss'].append(avg_loss)
        history['mse'].append(avg_mse)
        history['align'].append(avg_align)
        history['val_mse'].append(avg_val_mse)
        history['val_overlap'].append(avg_val_overlap)
        history['val_align'].append(avg_val_align)
        history['lr'].append(current_lr)
        
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} | Val MSE: {avg_val_mse:.4f} | Val Ov: {avg_val_overlap:.1f} | Val Align: {avg_val_align:.4f}")
        
        writer.writerow([epoch+1, avg_loss, avg_mse, avg_align, avg_val_mse, avg_val_overlap, avg_val_align, current_lr])
        log_file.flush()

    log_file.close()

    # 3. Save Model
    torch.save(model.state_dict(), "restorer_model.pth")
    print("Training finished. Model saved to restorer_model.pth")
    
    # 4. Plot Curves
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_mse'], label='Val MSE')
    plt.title('Loss')
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