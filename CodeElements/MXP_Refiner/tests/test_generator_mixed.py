import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.generator import SyntheticDataGenerator

def test_mixed_generation():
    gen = SyntheticDataGenerator(seed=42)
    # Mixed mode should produce a list of macros
    aligned, disturbed = gen.generate(count=30, mode='mixed', noise_level=10.0)
    
    assert len(aligned) == 30
    assert len(disturbed) == 30
    
    # Check basic properties
    for m in aligned:
        assert 'x' in m and 'y' in m
        
    # Mixed implies it might use different sub-generators.
    # We can check if it runs without error and produces valid output.
