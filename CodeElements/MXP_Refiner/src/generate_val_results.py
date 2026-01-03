import torch
import os
from tqdm import tqdm
from config import Config
from model import GraphToSeqRestorer
from evaluate_model import compute_metrics, get_graph_edges

def generate_results():
    print(f"Loading validation dataset from {Config.VAL_DATA_PATH}...")
    if not os.path.exists(Config.VAL_DATA_PATH):
        print(f"Error: {Config.VAL_DATA_PATH} not found.")
        return

    val_data_list = torch.load(Config.VAL_DATA_PATH, weights_only=False)
    print(f"Validation dataset loaded with {len(val_data_list)} samples.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    model = GraphToSeqRestorer(
        hidden_dim=Config.HIDDEN_DIM, 
        num_layers=Config.NUM_LAYERS, 
        num_heads=Config.NUM_HEADS
    ).to(device)
    
    if os.path.exists("restorer_model.pth"):
        model.load_state_dict(torch.load("restorer_model.pth", map_location=device, weights_only=True))
        print("Model loaded from restorer_model.pth")
    else:
        print("Warning: restorer_model.pth not found, using initialized weights.")
        
    model.eval()
    
    all_results = []
    
    print("Running inference on validation set...")
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_data_list)):
            data = data.to(device)
            
            # Extract original macros from info_dict (not tensors)
            aligned_macros = data.info_dict['aligned']
            disturbed_macros = data.info_dict['disturbed']
            category = data.info_dict.get('category', 'N/A')
            
            # Construct edge_attr_dict
            edge_attr_dict = {
                ('macro', 'phys_edge', 'macro'): data['macro', 'phys_edge', 'macro'].edge_attr,
                ('macro', 'logic_edge', 'macro'): data['macro', 'logic_edge', 'macro'].edge_attr,
                ('macro', 'align_edge', 'macro'): None 
            }
            
            # Run Inference
            pred_coords = model(data.x_dict, data.edge_index_dict, edge_attr_dict)
            
            # Convert pred_coords back to macro dicts
            restored_macros = []
            pred_coords_cpu = pred_coords.cpu().numpy()
            
            for idx, m in enumerate(disturbed_macros):
                new_m = m.copy()
                # Denormalize coordinates
                new_m['x'] = float(pred_coords_cpu[idx][0] * Config.CANVAS_WIDTH - new_m['w'] / 2)
                new_m['y'] = float(pred_coords_cpu[idx][1] * Config.CANVAS_HEIGHT - new_m['h'] / 2)
                restored_macros.append(new_m)
            
            # Compute Metrics
            metrics = compute_metrics(aligned_macros, disturbed_macros, restored_macros)
            
            # Get edges for visualization
            edges = get_graph_edges(disturbed_macros)
            
            all_results.append({
                'id': i,
                'metrics': metrics,
                'aligned': aligned_macros,
                'disturbed': disturbed_macros,
                'restored': restored_macros,
                'edges': edges,
                'category': category
            })
            
    output_path = os.path.join(Config.DATASET_DIR, "val_results.pt")
    print(f"Saving results to {output_path}")
    torch.save(all_results, output_path)
    print("Done.")

if __name__ == "__main__":
    generate_results()