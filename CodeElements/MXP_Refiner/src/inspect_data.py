from generator import SyntheticDataGenerator
from config import Config
from visualizer import GalleryGenerator
from data_builder import GraphBuilder
from restore_floorplan import FloorplanRestorationInference
import os
import numpy as np

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
    print(f"Generating data using mode: {Config.GENERATION_MODE}")
    generator = SyntheticDataGenerator(seed=Config.SEED, 
                                     canvas_width=Config.CANVAS_WIDTH, 
                                     canvas_height=Config.CANVAS_HEIGHT)
    
    # Initialize Restorer
    restorer = FloorplanRestorationInference()
    
    # We want "two kinds of macro size, each size has 10 macros" -> Total 20 macros, 2 clusters.
    count = 20
    
    data_groups = []
    
    # Generate a few independent groups (Reference layouts)
    for g in range(3):
        # Generate Reference
        aligned, _ = generator.generate(count=count, 
                                      mode=Config.GENERATION_MODE, 
                                      cluster_count=2,
                                      noise_level=0.0)
        
        # Build graph for reference
        ref_edges = get_graph_edges(aligned)
        
        # Generate Multiple Disturbed Samples from this Reference
        samples = []
        restored_list = []
        for s in range(2): # 2 samples per group for cleaner gallery
            # We use the internal helper to just perturb the existing reference
            disturbed = generator._perturb_macros(aligned, noise_level=Config.NOISE_LEVEL)
            
            # Restore
            restored = restorer.restore(disturbed)
            
            # Build graph for disturbed
            dist_edges = get_graph_edges(disturbed)
            # Build graph for restored
            rest_edges = get_graph_edges(restored)
            
            samples.append({
                'macros': disturbed,
                'edges': dist_edges
            })
            restored_list.append({
                'macros': restored,
                'edges': rest_edges
            })
            
        data_groups.append({
            'reference': {
                'macros': aligned,
                'edges': ref_edges
            },
            'samples': samples,
            'restored': restored_list
        })
    
    output_path = "data_preview.html"
    viz = GalleryGenerator(output_path)
    viz.generate(data_groups)
    
    print(f"Gallery preview generated at {output_path}")

if __name__ == "__main__":
    inspect_data()