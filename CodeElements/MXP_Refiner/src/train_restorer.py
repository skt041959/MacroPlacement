import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import os
import numpy as np
import logging
import time
import signal
import sys

from config import Config
from model import GraphUNet
from dataset import RestorationDataset
from evaluate_model import compute_metrics
from diffusion import DiffusionScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_restorer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")

    stop_training = False
    def signal_handler(sig, frame):
        nonlocal stop_training
        if not stop_training:
            logger.info("\nCtrl-C detected. Stopping after current epoch... (Press again to force stop)")
            stop_training = True
        else:
            logger.info("\nForce stopping...")
            sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)

    # 1. Dataset & Loader
    train_batch_size = Config.RESTORER_BATCH_SIZE * 4 # Default 32 * 4 = 128
    val_batch_size = Config.RESTORER_BATCH_SIZE * 8 # Default 32 * 8 = 256
    
    logger.info(f"Loading datasets from {Config.TRAIN_DATA_PATH} and {Config.VAL_DATA_PATH}")
    train_dataset = RestorationDataset(path=Config.TRAIN_DATA_PATH)
    val_dataset = RestorationDataset(path=Config.VAL_DATA_PATH)
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    logger.info(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # 2. Model, Optimizer, Loss, Scheduler
    logger.info(f"Initializing GraphUNet with depth={Config.NUM_LAYERS}, hidden_dim={Config.HIDDEN_DIM}")
    model = GraphUNet(
        hidden_dim=Config.HIDDEN_DIM, 
        num_layers=Config.NUM_LAYERS, 
        num_heads=Config.NUM_HEADS
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params:,}")
    
    diffusion = DiffusionScheduler()
    # Move scheduler tensors to device
    diffusion.sqrt_alphas_cumprod = diffusion.sqrt_alphas_cumprod.to(device)
    diffusion.sqrt_one_minus_alphas_cumprod = diffusion.sqrt_one_minus_alphas_cumprod.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.RESTORER_LR)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.RESTORER_EPOCHS)
    criterion = nn.MSELoss()

    # History for tracking
    history = {'loss': [], 'val_mse': [], 'val_overlap': [], 'val_align': [], 'lr': []}
    
    # Open CSV for structured logging
    csv_file = open("metrics.csv", "w", newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["Epoch", "Loss", "Val_MSE", "Val_Overlap", "Val_Align", "LR"])

    # Log noise distribution once to verify
    test_t = torch.tensor([0, 50, 99], device=device)
    test_x0 = torch.zeros((3, 2), device=device)
    test_noisy, test_noise = diffusion.add_noise(test_x0, test_t.repeat_interleave(1))
    logger.info(f"Diffusion Verify - t=[0, 50, 99] Noise Stds: {test_noisy[0].std().item():.4f}, {test_noisy[1].std().item():.4f}, {test_noisy[2].std().item():.4f}")

    for epoch in range(Config.RESTORER_EPOCHS):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0
        current_lr = optimizer.param_groups[0]['lr']
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.RESTORER_EPOCHS}", unit="batch")
        
        batch_idx = 0
        for data in pbar:
            data = data.to(device)
            
            # Sample random timesteps per graph
            t = torch.randint(0, diffusion.num_steps, (data.num_graphs,), device=device)
            # Expand t to node level
            t_expanded = t[data['macro'].batch]
            
            # Add noise to ground truth x0
            noisy_x, noise = diffusion.add_noise(data.y_coords, t_expanded)
            
            # Update x_dict with noisy coordinates
            x_dict = data.x_dict.copy()
            x_dict['macro'] = torch.cat([noisy_x, data.x_dict['macro'][:, 2:]], dim=-1)
            
            edge_attr_dict = {
                ('macro', 'phys_edge', 'macro'): data['macro', 'phys_edge', 'macro'].edge_attr,
                ('macro', 'logic_edge', 'macro'): data['macro', 'logic_edge', 'macro'].edge_attr,
                ('macro', 'align_edge', 'macro'): None 
            }
            
            optimizer.zero_grad()
            pred_noise = model(x_dict, data.edge_index_dict, edge_attr_dict, t)
            
            loss = criterion(pred_noise, noise)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 20 == 0:
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}", 
                    'LR': f"{current_lr:.6f}",
                    'Noise_Std': f"{pred_noise.std().item():.3f}"
                })
            batch_idx += 1
            
        # Validation
        val_start_time = time.time()
        model.eval()
        val_mse = 0
        val_overlap = 0
        val_align_recovery = 0
        total_graphs = 0
        
        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validating", leave=False):
                data = data.to(device)
                
                # Validation: mid-point noise prediction
                t = torch.full((data.num_graphs,), diffusion.num_steps // 2, device=device, dtype=torch.long)
                t_expanded = t[data['macro'].batch]
                noisy_x, noise = diffusion.add_noise(data.y_coords, t_expanded)
                
                x_dict = data.x_dict.copy()
                x_dict['macro'] = torch.cat([noisy_x, data.x_dict['macro'][:, 2:]], dim=-1)
                
                edge_attr_dict = {
                    ('macro', 'phys_edge', 'macro'): data['macro', 'phys_edge', 'macro'].edge_attr,
                    ('macro', 'logic_edge', 'macro'): data['macro', 'logic_edge', 'macro'].edge_attr,
                    ('macro', 'align_edge', 'macro'): None 
                }
                pred_noise = model(x_dict, data.edge_index_dict, edge_attr_dict, t)
                val_mse += criterion(pred_noise, noise).item() * data.num_graphs
                
                # Approximate restoration
                sqrt_alpha = diffusion.sqrt_alphas_cumprod[t_expanded].view(-1, 1)
                sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alphas_cumprod[t_expanded].view(-1, 1)
                pred_x0 = (noisy_x - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha
                
                # Vectorized metric preparation
                batch_vector = data['macro'].batch.cpu().numpy()
                feats = data.x_dict['macro'].cpu().numpy()
                pred_coords_np = pred_x0.cpu().numpy()
                target_coords_np = data.y_coords.cpu().numpy()
                
                for g_idx in range(data.num_graphs):
                    mask = (batch_vector == g_idx)
                    
                    # Layout-level arrays [x, y, w, h]
                    w_h = feats[mask, 2:] * np.array([Config.CANVAS_WIDTH, Config.CANVAS_HEIGHT])
                    
                    aligned = np.concatenate([
                        target_coords_np[mask] * np.array([Config.CANVAS_WIDTH, Config.CANVAS_HEIGHT]) - w_h/2,
                        w_h
                    ], axis=1)
                    
                    disturbed = np.concatenate([
                        feats[mask, :2] * np.array([Config.CANVAS_WIDTH, Config.CANVAS_HEIGHT]) - w_h/2,
                        w_h
                    ], axis=1)
                    
                    restored = np.concatenate([
                        pred_coords_np[mask] * np.array([Config.CANVAS_WIDTH, Config.CANVAS_HEIGHT]) - w_h/2,
                        w_h
                    ], axis=1)
                    
                    m = compute_metrics(aligned, disturbed, restored)
                    val_overlap += m['overlap_restored']
                    val_align_recovery += m['alignment_recovery']
                    
                total_graphs += data.num_graphs
        
        epoch_end_time = time.time()
        train_duration = val_start_time - epoch_start_time
        val_duration = epoch_end_time - val_start_time
        
        avg_val_mse = val_mse / total_graphs
        avg_val_overlap = val_overlap / total_graphs
        avg_val_align = val_align_recovery / total_graphs
        
        lr_scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        
        history['loss'].append(avg_loss)
        history['val_mse'].append(avg_val_mse)
        history['val_overlap'].append(avg_val_overlap)
        history['val_align'].append(avg_val_align)
        history['lr'].append(current_lr)
        
        logger.info(
            f"Epoch {epoch+1} Summary - Loss: {avg_loss:.4f} | Val MSE: {avg_val_mse:.6f} | "
            f"Val Ov: {avg_val_overlap:.1f} | Val Align: {avg_val_align:.4f} | "
            f"Time: {train_duration:.1f}s (train) + {val_duration:.1f}s (val)"
        )
        
        writer.writerow([epoch+1, avg_loss, avg_val_mse, avg_val_overlap, avg_val_align, current_lr])
        csv_file.flush()

        if stop_training:
            logger.info("Stopping training early.")
            break

    csv_file.close()
    torch.save(model.state_dict(), "restorer_model.pth")
    logger.info("Training finished. Model saved to restorer_model.pth")
    
    # Plot Curves
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.plot(history['loss'], label='Loss'); plt.title('Training Loss'); plt.legend()
    plt.subplot(1, 3, 2); plt.plot(history['val_overlap'], label='Overlap'); plt.title('Val Overlap'); plt.legend()
    plt.subplot(1, 3, 3); plt.plot(history['val_align'], label='Align'); plt.title('Val Align Recovery'); plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png')
    
    return history

if __name__ == "__main__":
    train_restorer()
