import numpy as np
from config import Config

def generate_random_data():
    # 1. Simulate data initialization (usually from DREAMPlace output)
    macros = [
        {'id': i, 'x': np.random.rand() * 800, 'y': np.random.rand() * 800, 
         'w': 50 + np.random.rand() * 50, 'h': 50 + np.random.rand() * 50} 
        for i in range(Config.MACRO_COUNT)
    ]
    # Randomly generate netlist
    netlist = [(np.random.randint(0, Config.MACRO_COUNT), np.random.randint(0, Config.MACRO_COUNT), 1.0) for _ in range(200)]
    return macros, netlist
