import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config

def test_config_data_generation():
    assert hasattr(Config, 'SEED')
    assert hasattr(Config, 'GENERATION_MODE')
    assert hasattr(Config, 'GRID_COLS')
