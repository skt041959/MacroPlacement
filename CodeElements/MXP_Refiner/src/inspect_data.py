from generator import SyntheticDataGenerator
from config import Config
from visualizer import GalleryGenerator
from data_builder import GraphBuilder
from restore_floorplan import FloorplanRestorationInference
import os
import numpy as np
import torch
import random

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

def inspect_data():
    print("Loading samples from categorized datasets...")
    
    data_groups = []
    restorer = FloorplanRestorationInference()
    
    # Pick 2 samples from each category
    for category in Config.CATEGORIES:
        path = Config.DATASET_PATH_TEMPLATE.format(category)
        if not os.path.exists(path): continue
        
        print(f"Sampling from {path}...")
        dataset = torch.load(path, weights_only=False)
        indices = random.sample(range(len(dataset)), 2)
        
        for idx in indices:
            data = dataset[idx]
            
            # Use stored macros for exact visualization
            aligned = data.aligned_macros
            disturbed = data.disturbed_macros
            
            # Restore
            restored = restorer.restore(disturbed)
            
            # Compute Metrics
            # (Adding dummy values if needed or computing properly)
            from evaluate_model import compute_metrics
            metrics = compute_metrics(aligned, disturbed, restored)
            
            # Edges
            ref_edges = get_graph_edges(aligned)
            dist_edges = get_graph_edges(disturbed)
            rest_edges = get_graph_edges(restored)
            
            data_groups.append({
                'reference': {'macros': aligned, 'edges': ref_edges},
                'samples': [{'macros': disturbed, 'edges': dist_edges}],
                'restored': [{'macros': restored, 'edges': rest_edges}],
                'metrics': metrics
            })
    
    output_path = "categorized_preview.html"
    viz = GalleryGenerator(output_path)
    viz.generate(data_groups)
    print(f"Categorized preview generated at {output_path}")

if __name__ == "__main__":
    inspect_data()