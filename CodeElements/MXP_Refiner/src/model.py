import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class HeteroGATEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads):
        super().__init__()
        
        self.node_encoder = Linear(4, hidden_dim)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('macro', 'phys_edge', 'macro'): GATv2Conv(
                    hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                    edge_dim=1, add_self_loops=False
                ),
                ('macro', 'align_edge', 'macro'): GATv2Conv(
                    hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                    add_self_loops=False
                ),
                ('macro', 'logic_edge', 'macro'): GATv2Conv(
                    hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                    edge_dim=1, add_self_loops=False
                ),
            }, aggr='sum')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x = self.node_encoder(x_dict['macro'])
        x_dict_copy = {key: val for key, val in x_dict.items()}
        x_dict_copy['macro'] = x

        for conv in self.convs:
            out_dict = conv(x_dict_copy, edge_index_dict, edge_attr_dict)
            x_dict_copy = {key: F.elu(x) for key, x in out_dict.items()}
        
        return x_dict_copy['macro']

class HeteroGATRefiner(nn.Module):
    def __init__(self, hidden_dim, out_dim, num_layers, num_heads):
        super().__init__()
        self.encoder = HeteroGATEncoder(hidden_dim, num_layers, num_heads)
        
        self.actor_head = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, out_dim)
        )
        
        self.critic_head = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, 1)
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        node_embed = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        
        global_embed = torch.mean(node_embed, dim=0) 
        
        action_logits = self.actor_head(node_embed)
        value = self.critic_head(global_embed)
        
        return action_logits, value

class FloorplanRestorer(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads):
        super().__init__()
        self.encoder = HeteroGATEncoder(hidden_dim, num_layers, num_heads)
        
        # Sequence decoder - using a simple GRU for now as suggested by spec
        # It takes the node embedding and outputs (x, y)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 2) # (x, y)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        node_embed = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        
        # We treat the nodes as a sequence based on their index
        # Shape: [1, N, hidden_dim] for GRU
        seq_input = node_embed.unsqueeze(0)
        
        output, _ = self.decoder(seq_input)
        
        # Output shape: [1, N, hidden_dim] -> [N, hidden_dim]
        output = output.squeeze(0)
        
        # Restored coordinates
        coords = self.output_layer(output)
        
        return coords

class GraphToSeqRestorer(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads, dropout=0.1):
        super().__init__()
        self.encoder = HeteroGATEncoder(hidden_dim, num_layers, num_heads)
        
        # Non-Autoregressive Transformer Configuration
        # We use the GNN embeddings as BOTH 'memory' (graph context) and 'tgt' (initial query)
        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=3) # Independent layers from GNN if desired
        
        self.output_layer = nn.Linear(hidden_dim, 2) # (x, y)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # 1. Encode Graph Structure
        # node_embed shape: [N, hidden_dim]
        node_embed = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        
        # 2. Prepare for Transformer
        # TransformerDecoder requires batch dimension for batch_first=True: [Batch, Seq, Dim]
        # In this single-graph setups, Batch=1.
        # Target (Queries) = Node Embeddings (Simulating "refinement" of the state)
        # Memory (Keys/Vals) = Node Embeddings (Access to graph topology info)
        
        # [N, D] -> [1, N, D]
        h = node_embed.unsqueeze(0)
        
        # 3. Parallel Decode (Non-Autoregressive)
        # No causal mask is used. All nodes attend to all other nodes.
        # tgt=h, memory=h
        out_seq = self.decoder(tgt=h, memory=h)
        
        # [1, N, D] -> [N, D]
        out_seq = out_seq.squeeze(0)
        
        # 4. Coordinate Projection
        coords = self.output_layer(out_seq)
        
        return coords
