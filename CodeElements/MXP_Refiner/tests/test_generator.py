import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.generator import SyntheticDataGenerator

def test_reproducibility():
    gen1 = SyntheticDataGenerator(seed=42)
    aligned1, disturbed1 = gen1.generate(count=10)
    
    gen2 = SyntheticDataGenerator(seed=42)
    aligned2, disturbed2 = gen2.generate(count=10)
    
    assert aligned1 == aligned2
    assert disturbed1 == disturbed2

def test_random_generation():
    gen = SyntheticDataGenerator(seed=123)
    count = 15
    aligned, disturbed = gen.generate(count=count, mode='random')
    
    assert len(aligned) == count
    assert len(disturbed) == count
    for m in aligned:
        assert 'x' in m and 'y' in m and 'w' in m and 'h' in m and 'id' in m
        assert m['x'] >= 0
        assert m['y'] >= 0

def test_grid_generation():
    gen = SyntheticDataGenerator()
    aligned, disturbed = gen.generate(count=4, mode='grid', grid_cols=2)
    
    assert len(aligned) == 4
    # Check if they are arranged in a grid-like manner
    # For a 2x2 grid, we expect some alignment in the ALIGNED set
    assert aligned[0]['x'] == aligned[2]['x']
    assert aligned[1]['x'] == aligned[3]['x']
    assert aligned[0]['y'] == aligned[1]['y']
    assert aligned[2]['y'] == aligned[3]['y']

def test_row_generation():
    gen = SyntheticDataGenerator()
    aligned, disturbed = gen.generate(count=5, mode='rows')
    
    assert len(aligned) == 5
    # All should be on same Y or aligned by rows
    y_coords = set(m['y'] for m in aligned)
    assert len(y_coords) < 5 # Should be some grouping

def test_perturbation():
    gen = SyntheticDataGenerator(seed=1)
    # Generate with 0 noise -> should be identical
    aligned, disturbed = gen.generate(count=5, mode='grid', noise_level=0.0)
    assert aligned == disturbed
    
    # Generate with noise
    aligned, disturbed = gen.generate(count=5, mode='grid', noise_level=10.0)
    # Positions should differ
    diff_count = 0
    for m1, m2 in zip(aligned, disturbed):
        if m1['x'] != m2['x'] or m1['y'] != m2['y']:
            diff_count += 1
    assert diff_count > 0
