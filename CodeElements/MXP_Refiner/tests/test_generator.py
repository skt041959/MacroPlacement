import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.generator import SyntheticDataGenerator

def test_reproducibility():
    gen1 = SyntheticDataGenerator(seed=42)
    data1 = gen1.generate(count=10)
    
    gen2 = SyntheticDataGenerator(seed=42)
    data2 = gen2.generate(count=10)
    
    assert data1 == data2

def test_random_generation():
    gen = SyntheticDataGenerator(seed=123)
    count = 15
    data = gen.generate(count=count, mode='random')
    
    assert len(data) == count
    for m in data:
        assert 'x' in m and 'y' in m and 'w' in m and 'h' in m and 'id' in m
        assert m['x'] >= 0
        assert m['y'] >= 0

def test_grid_generation():
    gen = SyntheticDataGenerator()
    data = gen.generate(count=4, mode='grid', grid_cols=2)
    
    assert len(data) == 4
    # Check if they are arranged in a grid-like manner
    # For a 2x2 grid, we expect some alignment
    assert data[0]['x'] == data[2]['x']
    assert data[1]['x'] == data[3]['x']
    assert data[0]['y'] == data[1]['y']
    assert data[2]['y'] == data[3]['y']

def test_row_generation():
    gen = SyntheticDataGenerator()
    data = gen.generate(count=5, mode='rows')
    
    assert len(data) == 5
    # All should be on same Y or aligned by rows
    # Implementation dependent, but should be distinct from random
    # Check if all have same y (if single row)
    y_coords = set(m['y'] for m in data)
    assert len(y_coords) < 5 # Should be some grouping
