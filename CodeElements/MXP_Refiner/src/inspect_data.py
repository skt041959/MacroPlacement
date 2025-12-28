from utils import generate_random_data
from visualizer import DashboardGenerator
import os

def inspect_data():
    print("Generating random data...")
    macros, netlist = generate_random_data()
    
    # We want to show a single static frame of the data
    # DashboardGenerator expects list of snapshots (history)
    snapshots = [macros]
    
    # Create a dummy history just to satisfy the generator
    history = {
        'rewards': [0],
        'losses': [0]
    }
    
    output_path = "data_preview.html"
    viz = DashboardGenerator(output_path)
    viz.generate(history, snapshots, netlist)
    
    print(f"Data preview generated at {output_path}")

if __name__ == "__main__":
    inspect_data()
