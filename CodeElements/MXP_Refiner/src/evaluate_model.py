from config import Config
from restore_floorplan import FloorplanRestorationInference
from data_builder import GraphBuilder
from geometry import calculate_total_overlap, calculate_alignment_recovery
import os
import torch
import numpy as np
from tqdm import tqdm

def compute_metrics(aligned, disturbed, restored):
    """
    Computes core metrics for floorplan restoration.
    aligned, disturbed, restored: List of macro dicts {'x', 'y', 'w', 'h'}
    """
    # 1. MSE (Centers)
    sq_dist = 0
    for i in range(len(aligned)):
        sq_dist += (aligned[i]['x'] - restored[i]['x'])**2 + (aligned[i]['y'] - restored[i]['y'])**2
    mse = sq_dist / len(aligned)
    
    # 2. Overlap
    ov_aligned = calculate_total_overlap(aligned)
    ov_disturbed = calculate_total_overlap(disturbed)
    ov_restored = calculate_total_overlap(restored)
    
    # 3. Alignment Recovery
    recovery = calculate_alignment_recovery(aligned, disturbed, restored)
    
    return {
        'mse': float(mse),
        'overlap_aligned': float(ov_aligned),
        'overlap_disturbed': float(ov_disturbed),
        'overlap_restored': float(ov_restored),
        'alignment_recovery': float(recovery)
    }

def get_graph_edges(macros):
    """
    Helper to extract edges from a set of macros for analysis.
    """
    netlist = [] 
    builder = GraphBuilder(macros, netlist)
    graph = builder.build_hetero_graph()
    edges = {}
    if ('macro', 'phys_edge', 'macro') in graph.edge_index_dict:
        edges['phys'] = graph.edge_index_dict[('macro', 'phys_edge', 'macro')].t().tolist()
    if ('macro', 'align_edge', 'macro') in graph.edge_index_dict:
        edges['align'] = graph.edge_index_dict[('macro', 'align_edge', 'macro')].t().tolist()
    return edges

def evaluate_full_val_set():
    """
    Evaluates the restoration model on the entire validation set and prints statistics.
    """
    print(f"Loading validation dataset from {Config.VAL_DATA_PATH}...")
    if not os.path.exists(Config.VAL_DATA_PATH):
        print(f"Error: {Config.VAL_DATA_PATH} not found.")
        return

    val_data_list = torch.load(Config.VAL_DATA_PATH, weights_only=False)
    print(f"Validation dataset loaded with {len(val_data_list)} samples.")
    
    restorer = FloorplanRestorationInference()
    all_metrics = []
    
    print("Running inference on validation set...")
    for data in tqdm(val_data_list):
        aligned_macros = data.info_dict['aligned']
        disturbed_macros = data.info_dict['disturbed']
            
        # Run Inference
        restored_macros = restorer.restore(disturbed_macros)
        
        # Compute Metrics
        metrics = compute_metrics(aligned_macros, disturbed_macros, restored_macros)
        all_metrics.append(metrics)

    # Statistics
    mses = [m['mse'] for m in all_metrics]
    recoveries = [m['alignment_recovery'] for m in all_metrics]
    overlaps_disturbed = [m['overlap_disturbed'] for m in all_metrics]
    overlaps_restored = [m['overlap_restored'] for m in all_metrics]
    
    print("\n--- Validation Statistics ---")
    print(f"Mean MSE: {np.mean(mses):.6f}")
    print(f"Max MSE: {np.max(mses):.6f}")
    print(f"Mean Alignment Recovery: {np.mean(recoveries)*100:.2f}%")
    print(f"Min Alignment Recovery: {np.min(recoveries)*100:.2f}%")
    print(f"Mean Overlap (Disturbed): {np.mean(overlaps_disturbed):.2f}")
    print(f"Mean Overlap (Restored): {np.mean(overlaps_restored):.2f}")
    print(f"Overlap Reduction: {(1 - np.mean(overlaps_restored)/(np.mean(overlaps_disturbed)+1e-6))*100:.2f}%")

if __name__ == "__main__":
    evaluate_full_val_set()
