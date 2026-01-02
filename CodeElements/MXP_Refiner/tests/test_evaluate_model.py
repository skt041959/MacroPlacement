import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from evaluate_model import compute_metrics
from geometry import calculate_total_overlap, calculate_alignment_recovery

def test_compute_metrics():
    # Setup dummy data
    aligned = [
        {'id': 0, 'x': 0.0, 'y': 0.0, 'w': 10.0, 'h': 10.0},
        {'id': 1, 'x': 10.0, 'y': 0.0, 'w': 10.0, 'h': 10.0}
    ]
    disturbed = [
        {'id': 0, 'x': 1.0, 'y': 1.0, 'w': 10.0, 'h': 10.0},
        {'id': 1, 'x': 9.0, 'y': 1.0, 'w': 10.0, 'h': 10.0}
    ]
    restored = [
        {'id': 0, 'x': 0.5, 'y': 0.5, 'w': 10.0, 'h': 10.0},
        {'id': 1, 'x': 10.5, 'y': 0.5, 'w': 10.0, 'h': 10.0}
    ]
    
    metrics = compute_metrics(aligned, disturbed, restored)
    
    # Check keys
    assert 'mse' in metrics
    assert 'overlap_aligned' in metrics
    assert 'overlap_disturbed' in metrics
    assert 'overlap_restored' in metrics
    assert 'alignment_recovery' in metrics
    
    # Check values logic
    # MSE: ((0-0.5)^2 + (0-0.5)^2 + (10-10.5)^2 + (0-0.5)^2) / 2
    # = (0.25 + 0.25 + 0.25 + 0.25) / 2 = 1.0 / 2 = 0.5
    assert metrics['mse'] == pytest.approx(0.5)
    
    # Overlap Aligned: 0 (touching but not overlapping)
    assert metrics['overlap_aligned'] == 0.0
    
    # Overlap Disturbed: 
    # Box 1: [1, 11] x [1, 11]
    # Box 2: [9, 19] x [1, 11]
    # X overlap: [9, 11] -> width 2
    # Y overlap: [1, 11] -> height 10
    # Area = 20
    assert metrics['overlap_disturbed'] == pytest.approx(20.0)
    
    # Overlap Restored:
    # Box 1: [0.5, 10.5] x [0.5, 10.5]
    # Box 2: [10.5, 20.5] x [0.5, 10.5]
    # X overlap: 0 (touching at 10.5)
    assert metrics['overlap_restored'] == 0.0
