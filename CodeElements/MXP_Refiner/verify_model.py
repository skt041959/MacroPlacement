
import torch
from src.model import GraphToSeqRestorer
from src.config import Config

def verify_model():
    print("Verifying GraphToSeqRestorer...")
    
    # Dummy Config
    hidden_dim = 64
    num_vals = 10 # num macros
    
    model = GraphToSeqRestorer(hidden_dim, num_layers=2, num_heads=4)
    print("Model instantiated.")
    
    # Dummy Inputs
    # x_dict: {'macro': [N, 4]} (assuming encoder expects 4 input features as per Linear(4, hidden))
    x_dict = {'macro': torch.randn(num_vals, 4)}
    
    # edge_index_dict: dummy edges
    edge_index = torch.randint(0, num_vals, (2, 20))
    edge_index_dict = {
        ('macro', 'phys_edge', 'macro'): edge_index,
        ('macro', 'align_edge', 'macro'): edge_index,
        ('macro', 'logic_edge', 'macro'): edge_index
    }
    
    # edge_attr_dict (GATv2Conv with edge_dim=1 expects edge features of dim 1)
    edge_attr = torch.randn(20, 1)
    edge_attr_dict = {
        ('macro', 'phys_edge', 'macro'): edge_attr,
        # align_edge has no edge_dim in the model definition?
        # Let's check model.py:
        # ('macro', 'align_edge', 'macro'): GATv2Conv(..., edge_dim=None/Default) -> NO edge_dim param passed for align_edge?
        # Wait, let's look at model.py:
        # ('macro', 'align_edge', 'macro'): GATv2Conv(..., add_self_loops=False) -> No edge_dim specified, so it expects None or no edge_attr.
        # ('macro', 'logic_edge', 'macro'): GATv2Conv(..., edge_dim=1)
        ('macro', 'logic_edge', 'macro'): edge_attr
    }
    
    # Forward
    try:
        coords = model(x_dict, edge_index_dict, edge_attr_dict)
        print(f"Output shape: {coords.shape}")
        
        if coords.shape == (num_vals, 2):
            print("SUCCESS: Output shape matches expected [N, 2].")
        else:
            print(f"FAILURE: Expected [{num_vals}, 2], got {coords.shape}")
            
    except Exception as e:
        print(f"FAILURE: Runtime error during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_model()
