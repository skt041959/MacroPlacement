import pytest
import sys
import os

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.geometry import calculate_overlap_area

def test_no_overlap():
    m1 = {'x': 0, 'y': 0, 'w': 10, 'h': 10}
    m2 = {'x': 20, 'y': 20, 'w': 10, 'h': 10}
    assert calculate_overlap_area(m1, m2) == 0.0

def test_full_overlap():
    m1 = {'x': 0, 'y': 0, 'w': 10, 'h': 10}
    m2 = {'x': 0, 'y': 0, 'w': 10, 'h': 10}
    assert calculate_overlap_area(m1, m2) == 100.0

def test_partial_overlap():
    m1 = {'x': 0, 'y': 0, 'w': 10, 'h': 10}
    m2 = {'x': 5, 'y': 5, 'w': 10, 'h': 10}
    # Overlap is from (5,5) to (10,10), so 5x5 = 25
    assert calculate_overlap_area(m1, m2) == 25.0

def test_touching_no_overlap():
    m1 = {'x': 0, 'y': 0, 'w': 10, 'h': 10}
    m2 = {'x': 10, 'y': 0, 'w': 10, 'h': 10}
    assert calculate_overlap_area(m1, m2) == 0.0

def test_containment():
    m1 = {'x': 0, 'y': 0, 'w': 20, 'h': 20}
    m2 = {'x': 5, 'y': 5, 'w': 5, 'h': 5}
    assert calculate_overlap_area(m1, m2) == 25.0

def test_total_overlap():
    from src.geometry import calculate_total_overlap
    macros = [
        {'x': 0, 'y': 0, 'w': 10, 'h': 10},
        {'x': 5, 'y': 0, 'w': 10, 'h': 10},
        {'x': 20, 'y': 20, 'w': 10, 'h': 10}
    ]
    # m0 and m1 overlap by 5x10 = 50
    # m0 and m2: 0
    # m1 and m2: 0
    assert calculate_total_overlap(macros) == 50.0

def test_alignment_score_perfect():
    from src.geometry import calculate_alignment_score
    macros = [
        {'x': 0, 'y': 0, 'w': 10, 'h': 10},
        {'x': 0, 'y': 20, 'w': 10, 'h': 10}, # X aligned
        {'x': 20, 'y': 20, 'w': 10, 'h': 10} # Y aligned with m1
    ]
    # m0 and m1 aligned on X
    # m1 and m2 aligned on Y
    # Total score should be > 0
    score = calculate_alignment_score(macros)
    assert score > 0

def test_alignment_score_none():
    from src.geometry import calculate_alignment_score
    macros = [
        {'x': 0, 'y': 0, 'w': 10, 'h': 10},
        {'x': 13, 'y': 27, 'w': 10, 'h': 10}
    ]
    assert calculate_alignment_score(macros) == 0.0

def test_grid_alignment():
    from src.geometry import calculate_alignment_score
    macros = [
        {'x': 10, 'y': 10, 'w': 10, 'h': 10},
        {'x': 20, 'y': 20, 'w': 10, 'h': 10}
    ]
    # Both are on 10-unit grid
    score = calculate_alignment_score(macros, grid_size=10.0)
    assert score > 0

def test_alignment_recovery():
    from src.geometry import calculate_alignment_recovery
    aligned = [{'x': 0, 'y': 0, 'w': 10, 'h': 10}, {'x': 0, 'y': 20, 'w': 10, 'h': 10}]
    disturbed = [{'x': 2, 'y': 1, 'w': 10, 'h': 10}, {'x': -1, 'y': 19, 'w': 10, 'h': 10}]
    restored = [{'x': 0.1, 'y': 0.1, 'w': 10, 'h': 10}, {'x': 0.1, 'y': 20.1, 'w': 10, 'h': 10}]
    
    recovery = calculate_alignment_recovery(aligned, disturbed, restored)
    # restored is much closer to aligned than disturbed is
    assert recovery > 0.5
    assert recovery <= 1.0
