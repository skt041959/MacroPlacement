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
    O(N^2) implementation.
    """
    total_overlap = 0.0
    n = len(macros)
    for i in range(n):
        for j in range(i + 1, n):
            total_overlap += calculate_overlap_area(macros[i], macros[j])
    return total_overlap

def calculate_alignment_score(macros, grid_size=None, threshold=0.1):
    """
    Calculates an alignment score for a list of macros.
    Score increases with the number of aligned edges.
    """
    score = 0.0
    n = len(macros)
    
    # 1. Edge alignment (pairwise)
    for i in range(n):
        for j in range(i + 1, n):
            # Check X alignment (left edges)
            if abs(macros[i]['x'] - macros[j]['x']) < threshold:
                score += 1.0
            # Check Y alignment (bottom edges)
            if abs(macros[i]['y'] - macros[j]['y']) < threshold:
                score += 1.0
                
    # 2. Grid alignment
    if grid_size is not None:
        for m in macros:
            # Check x alignment to grid
            x_mod = m['x'] % grid_size
            if x_mod < threshold or abs(grid_size - x_mod) < threshold:
                score += 1.0
            # Check y alignment to grid
            y_mod = m['y'] % grid_size
            if y_mod < threshold or abs(grid_size - y_mod) < threshold:
                score += 1.0
                
    return score

def calculate_alignment_recovery(aligned, disturbed, restored, threshold=0.1):
    """
    Calculates how much of the original alignment was recovered.
    Returns a value between 0 and 1 (usually).
    """
    s_aligned = calculate_alignment_score(aligned, threshold=threshold)
    s_disturbed = calculate_alignment_score(disturbed, threshold=threshold)
    s_restored = calculate_alignment_score(restored, threshold=threshold)
    
    denom = s_aligned - s_disturbed
    if abs(denom) < 1e-6:
        return 1.0 if abs(s_restored - s_aligned) < 1e-6 else 0.0
        
    recovery = (s_restored - s_disturbed) / denom
    return float(np.clip(recovery, 0.0, 1.0))
