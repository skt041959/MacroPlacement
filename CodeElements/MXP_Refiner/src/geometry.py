import numpy as np
import torch

def batch_knn_edges(x, batch, k=5):
    """
    Computes K-Nearest Neighbors edges for a batch of graphs.
    x: [N, D] node features (coordinates)
    batch: [N] batch indices
    k: Number of neighbors
    Returns: edge_index [2, E], edge_attr [E, 1]
    """
    # Create unique batch offsets to process all graphs in parallel without mixing
    # But for simplicity with variable graph sizes, a loop over batch items is often safer/easier
    # unless using torch_cluster.knn_graph (which we assume isn't available)
    
    device = x.device
    num_graphs = batch.max().item() + 1
    
    src_list = []
    dst_list = []
    dists_list = []
    
    for i in range(num_graphs):
        mask = (batch == i)
        if not mask.any(): continue
        
        # Local indices in the batch
        nodes = x[mask]
        num_nodes = nodes.size(0)
        
        # If graph is too small for k, adjust k
        curr_k = min(k, num_nodes - 1)
        if curr_k <= 0: continue
            
        # Compute Pairwise Distance
        # dists: [num_nodes, num_nodes]
        dists = torch.cdist(nodes, nodes)
        
        # Get Top-K (smallest distances)
        # We skip the 0-th element (self-loop with dist 0)
        vals, indices = torch.topk(dists, k=curr_k + 1, dim=1, largest=False)
        
        vals = vals[:, 1:] # Drop self [N, K]
        indices = indices[:, 1:] # [N, K]
        
        # Global Node Indices
        # We need to map local indices 0..N-1 back to global indices
        # find indices where mask is True
        global_indices_map = torch.nonzero(mask).squeeze()
        
        # Source: [N, K] -> [N*K]
        # Repeat global indices for source
        src_local = global_indices_map.unsqueeze(1).repeat(1, curr_k)
        
        # Target: [N, K] -> mapped to global
        dst_local = global_indices_map[indices]
        
        src_list.append(src_local.flatten())
        dst_list.append(dst_local.flatten())
        dists_list.append(vals.flatten())
        
    if not src_list:
        return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty((0, 1), dtype=torch.float, device=device)
        
    edge_index = torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)
    edge_attr = torch.cat(dists_list).unsqueeze(1)
    
    return edge_index, edge_attr

def calculate_overlap_area(m1, m2):
    """
    Calculates the overlap area between two macros.
    m1, m2: dict with 'x', 'y', 'w', 'h'
    """
    dx = min(m1['x'] + m1['w'], m2['x'] + m2['w']) - max(m1['x'], m2['x'])
    dy = min(m1['y'] + m1['h'], m2['y'] + m2['h']) - max(m1['y'], m2['y'])
    
    if dx > 0 and dy > 0:
        return float(dx * dy)
    return 0.0

def calculate_total_overlap(macros):
    """
    Calculates the total overlap area among a list of macros.
    Optimized O(N^2) using NumPy vectorization.
    Supports either list of dicts or a single (N, 4) NumPy array [x, y, w, h].
    """
    if isinstance(macros, np.ndarray):
        x, y, w, h = macros[:, 0], macros[:, 1], macros[:, 2], macros[:, 3]
    else:
        n = len(macros)
        if n < 2: return 0.0
        x = np.array([m['x'] for m in macros])
        y = np.array([m['y'] for m in macros])
        w = np.array([m['w'] for m in macros])
        h = np.array([m['h'] for m in macros])
    
    n = len(x)
    if n < 2: return 0.0
    
    x2 = x + w
    y2 = y + h
    
    total_overlap = 0.0
    for i in range(n):
        dx = np.minimum(x2[i], x2[i+1:]) - np.maximum(x[i], x[i+1:])
        dy = np.minimum(y2[i], y2[i+1:]) - np.maximum(y[i], y[i+1:])
        overlap_areas = np.maximum(0, dx) * np.maximum(0, dy)
        total_overlap += np.sum(overlap_areas)
        
    return float(total_overlap)

def calculate_alignment_score(macros, grid_size=None, threshold=0.1):
    """
    Calculates an alignment score for a list of macros.
    Optimized with NumPy.
    Supports either list of dicts or a single (N, 2) NumPy array [x, y].
    """
    if isinstance(macros, np.ndarray):
        x, y = macros[:, 0], macros[:, 1]
    else:
        n = len(macros)
        if n < 1: return 0.0
        x = np.array([m['x'] for m in macros])
        y = np.array([m['y'] for m in macros])
    
    n = len(x)
    if n < 1: return 0.0
    
    score = 0.0
    if n > 1:
        diff_x = np.abs(x[:, np.newaxis] - x[np.newaxis, :])
        score += np.sum(diff_x[np.triu_indices(n, k=1)] < threshold)
        diff_y = np.abs(y[:, np.newaxis] - y[np.newaxis, :])
        score += np.sum(diff_y[np.triu_indices(n, k=1)] < threshold)
                
    if grid_size is not None:
        x_mod = x % grid_size
        score += np.sum(np.minimum(x_mod, grid_size - x_mod) < threshold)
        y_mod = y % grid_size
        score += np.sum(np.minimum(y_mod, grid_size - y_mod) < threshold)
                
    return float(score)

def calculate_alignment_recovery(aligned, disturbed, restored, threshold=0.1):
    """
    Calculates how much of the original alignment was recovered.
    Supports either list of dicts or NumPy arrays.
    """
    s_aligned = calculate_alignment_score(aligned, threshold=threshold)
    s_disturbed = calculate_alignment_score(disturbed, threshold=threshold)
    s_restored = calculate_alignment_score(restored, threshold=threshold)
    
    denom = s_aligned - s_disturbed
    if abs(denom) < 1e-6:
        return 1.0 if abs(s_restored - s_aligned) < 1e-6 else 0.0
        
    recovery = (s_restored - s_disturbed) / denom
    return float(np.clip(recovery, 0.0, 1.0))

def alignment_energy_function(coords, align_edge_index, grid_pitch=50.0, lambda_grid=0.5, lambda_channel=0.5):
    """
    Computes a differentiable energy function for alignment guidance.
    Lower energy = better alignment.
    
    coords: [N, 2] tensor (requires_grad=True)
    align_edge_index: [2, E] tensor, indices of alignment edges
    grid_pitch: float, target grid spacing
    
    Returns: scalar energy tensor
    """
    # 1. Grid Alignment Energy (Snap-to-Grid)
    # E_grid = sum(sin^2(pi * x / K))
    # Using sin^2 creates minima at integer multiples of K
    
    # Scale coords to be relative to pitch
    scaled_coords = coords / grid_pitch
    sin_term = torch.sin(np.pi * scaled_coords)
    e_grid = torch.sum(sin_term ** 2)
    
    # 2. Channel Alignment Energy (Relative Alignment)
    # E_channel = sum(||x_i - x_j||^2) for aligned pairs
    if align_edge_index is not None and align_edge_index.size(1) > 0:
        src, dst = align_edge_index
        # Get coordinates of connected nodes
        p1 = coords[src]
        p2 = coords[dst]
        
        # We only care about aligning either X or Y, not necessarily Euclidean distance
        # But generally, aligning edges implies they should be close in one dimension
        # or exactly aligned. For simplicty in 'alignment edges' (which usually connect 
        # macros that should be aligned), minimizing L2 distance acts as a strong prior.
        # Alternatively, we could minimize min(|dx|, |dy|), but L2 is smoother.
        
        diff = p1 - p2
        e_channel = torch.sum(diff ** 2)
    else:
        e_channel = torch.tensor(0.0, device=coords.device)
        
    total_energy = lambda_grid * e_grid + lambda_channel * e_channel
    return total_energy
