import numpy as np

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
