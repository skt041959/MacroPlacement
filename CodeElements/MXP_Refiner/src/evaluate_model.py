import torch
import random
import numpy as np
from config import Config
from dataset import RestorationDataset
from restore_floorplan import FloorplanRestorationInference
from visualizer import GalleryGenerator
from data_builder import GraphBuilder

def get_graph_edges(macros):
    # Dummy netlist for now
    netlist = [] 
    
    builder = GraphBuilder(macros, netlist)
    graph = builder.build_hetero_graph()
    
    edges = {}
    
    # Extract Physical Edges
    if ('macro', 'phys_edge', 'macro') in graph.edge_index_dict:
        edge_index = graph.edge_index_dict[('macro', 'phys_edge', 'macro')]
        edges['phys'] = edge_index.t().tolist()
        
    # Extract Align Edges
    if ('macro', 'align_edge', 'macro') in graph.edge_index_dict:
        edge_index = graph.edge_index_dict[('macro', 'align_edge', 'macro')]
        edges['align'] = edge_index.t().tolist()
        
    # Extract Logic Edges
    if ('macro', 'logic_edge', 'macro') in graph.edge_index_dict:
        edge_index = graph.edge_index_dict[('macro', 'logic_edge', 'macro')]
        edges['logic'] = edge_index.t().tolist()
        
    return edges

def evaluate_on_samples(num_samples=5):
    print(f"Loading dataset from {Config.DATASET_PATH}...")
    # Load dataset (this might take a moment if it loads all into memory, but it's loaded as list)
    dataset = RestorationDataset(path=Config.DATASET_PATH)
    total_samples = len(dataset)
    print(f"Dataset loaded with {total_samples} samples.")
    
    # Select random indices
    indices = random.sample(range(total_samples), num_samples)
    print(f"Selected sample indices: {indices}")
    
    # Initialize Restorer
    restorer = FloorplanRestorationInference()
    
    data_groups = []
    
    for idx in indices:
        # The dataset returns HeteroData object with y_coords (target)
        # But we need the original macro dicts to reconstruct layout for visualization
        # RestorationDataset stores data as HeteroData list if loaded from disk.
        # It does NOT store the original macro dictionaries in the HeteroData object by default unless we hacked it.
        # Wait, the current `RestorationDataset` implementation:
        # self.data_list.append(data)
        # `data` is HeteroData. It has features x, edge_index, etc.
        # It does NOT have the original 'w', 'h', 'id' which are needed for visualization!
        
        # CRITICAL: We need to reconstruct macro dicts from HeteroData features.
        # Feature format: [cx/W, cy/H, w/W, h/H]
        data = dataset[idx]
        
        disturbed_macros = []
        feature_matrix = data['macro'].x
        
        for i in range(feature_matrix.shape[0]):
            norm_cx = feature_matrix[i, 0].item()
            norm_cy = feature_matrix[i, 1].item()
            norm_w = feature_matrix[i, 2].item()
            norm_h = feature_matrix[i, 3].item()
            
            w = norm_w * Config.CANVAS_WIDTH
            h = norm_h * Config.CANVAS_HEIGHT
            cx = norm_cx * Config.CANVAS_WIDTH
            cy = norm_cy * Config.CANVAS_HEIGHT
            
            x = cx - w/2
            y = cy - h/2
            
            disturbed_macros.append({
                'id': i,
                'x': x,
                'y': y,
                'w': w,
                'h': h
            })
            
        # Reconstruct Target (Aligned) Macros
        # data.y_coords is [N, 2] (cx, cy normalized)
        aligned_macros = []
        target_coords = data.y_coords
        for i in range(target_coords.shape[0]):
            norm_cx = target_coords[i, 0].item()
            norm_cy = target_coords[i, 1].item()
            
            # Dimensions are same as disturbed
            w = disturbed_macros[i]['w']
            h = disturbed_macros[i]['h']
            
            cx = norm_cx * Config.CANVAS_WIDTH
            cy = norm_cy * Config.CANVAS_HEIGHT
            
            x = cx - w/2
            y = cy - h/2
            
            aligned_macros.append({
                'id': i,
                'x': x,
                'y': y,
                'w': w,
                'h': h
            })
            
        # Run Inference
        restored_macros = restorer.restore(disturbed_macros)
        
        # Build Edges for visualization
        ref_edges = get_graph_edges(aligned_macros)
        dist_edges = get_graph_edges(disturbed_macros)
        rest_edges = get_graph_edges(restored_macros)
        
        # Structure for GalleryGenerator
        # One group per sample
        data_groups.append({
            'reference': {
                'macros': aligned_macros,
                'edges': ref_edges
            },
            'samples': [{
                'macros': disturbed_macros,
                'edges': dist_edges
            }],
            'restored': [{
                'macros': restored_macros,
                'edges': rest_edges
            }]
        })

    output_path = "evaluation_report.html"
    viz = GalleryGenerator(output_path)
    viz.generate(data_groups)
    
    print(f"Evaluation report generated at {output_path}")

if __name__ == "__main__":
    evaluate_on_samples()
