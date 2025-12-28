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
