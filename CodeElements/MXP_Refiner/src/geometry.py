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
