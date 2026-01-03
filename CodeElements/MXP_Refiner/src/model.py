import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear, TopKPooling
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from torch_geometric.utils import subgraph

class GraphUNet(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads, pool_ratio=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.node_encoder = Linear(4, hidden_dim)
        
        # Encoder Path
        self.encoder_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('macro', 'phys_edge', 'macro'): GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, edge_dim=1, add_self_loops=False),
                ('macro', 'align_edge', 'macro'): GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, add_self_loops=False),
                ('macro', 'logic_edge', 'macro'): GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, edge_dim=1, add_self_loops=False),
            }, aggr='sum')
            self.encoder_convs.append(conv)
            self.pools.append(TopKPooling(hidden_dim, ratio=pool_ratio))
            
        # Bottleneck
        self.bottleneck = HeteroConv({
            ('macro', 'phys_edge', 'macro'): GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, edge_dim=1, add_self_loops=False),
            ('macro', 'align_edge', 'macro'): GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, add_self_loops=False),
            ('macro', 'logic_edge', 'macro'): GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, edge_dim=1, add_self_loops=False),
        }, aggr='sum')
        
        # Decoder Path
        self.decoder_convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('macro', 'phys_edge', 'macro'): GATv2Conv(2 * hidden_dim, hidden_dim // num_heads, heads=num_heads, edge_dim=1, add_self_loops=False),
                ('macro', 'align_edge', 'macro'): GATv2Conv(2 * hidden_dim, hidden_dim // num_heads, heads=num_heads, add_self_loops=False),
                ('macro', 'logic_edge', 'macro'): GATv2Conv(2 * hidden_dim, hidden_dim // num_heads, heads=num_heads, edge_dim=1, add_self_loops=False),
            }, aggr='sum')
            self.decoder_convs.append(conv)
            
        self.output_layer = nn.Linear(hidden_dim, 2)

    def _filter_edge_dict(self, edge_index_dict, edge_attr_dict, subset, num_nodes):
        new_edge_index_dict = {}
        new_edge_attr_dict = {}
        
        for edge_type, edge_index in edge_index_dict.items():
            attr = edge_attr_dict.get(edge_type)
            # We must specify num_nodes so index_to_mask knows the dimension
            new_idx, new_attr = subgraph(subset, edge_index, edge_attr=attr, relabel_nodes=True, num_nodes=num_nodes)
            new_edge_index_dict[edge_type] = new_idx
            if attr is not None:
                new_edge_attr_dict[edge_type] = new_attr
            else:
                new_edge_attr_dict[edge_type] = None
                
        return new_edge_index_dict, new_edge_attr_dict

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x = self.node_encoder(x_dict['macro'])
        num_original_nodes = x.size(0)
        batch = x_dict.get('batch', torch.zeros(num_original_nodes, dtype=torch.long, device=x.device))
        
        xs = []
        perms = []
        edge_dicts = []
        attr_dicts = []
        
        # Encoder
        curr_x_dict = {'macro': x}
        curr_edge_index_dict = edge_index_dict
        curr_edge_attr_dict = edge_attr_dict
        curr_num_nodes = num_original_nodes
        
        for i in range(self.num_layers):
            out_dict = self.encoder_convs[i](curr_x_dict, curr_edge_index_dict, curr_edge_attr_dict)
            x = F.elu(out_dict['macro'])
            xs.append(x)
            edge_dicts.append(curr_edge_index_dict)
            attr_dicts.append(curr_edge_attr_dict)
            
            # Pooling (gPool)
            phys_idx = curr_edge_index_dict[('macro', 'phys_edge', 'macro')]
            phys_attr = curr_edge_attr_dict.get(('macro', 'phys_edge', 'macro'))
            
            x, _, _, batch, perm, score = self.pools[i](x, phys_idx, phys_attr, batch)
            perms.append(perm)
            
            # Filter all edges for the pooled graph
            curr_edge_index_dict, curr_edge_attr_dict = self._filter_edge_dict(curr_edge_index_dict, curr_edge_attr_dict, perm, curr_num_nodes)
            curr_num_nodes = x.size(0)
            curr_x_dict = {'macro': x}
            
        # Bottleneck
        out_dict = self.bottleneck(curr_x_dict, curr_edge_index_dict, curr_edge_attr_dict)
        x = F.elu(out_dict['macro'])
        
        # Decoder
        for i in range(self.num_layers - 1, -1, -1):
            # Unpooling (gUnpool)
            res = torch.zeros(xs[i].size(0), self.hidden_dim, device=x.device)
            res[perms[i]] = x
            x = res
            
            # Skip Connection (Concat)
            x = torch.cat([x, xs[i]], dim=-1)
            
            curr_x_dict = {'macro': x}
            # Use the edge dict from the corresponding encoder level (unpooled edges)
            out_dict = self.decoder_convs[i](curr_x_dict, edge_dicts[i], attr_dicts[i])
            x = F.elu(out_dict['macro'])
            curr_x_dict = {'macro': x}
            
        return self.output_layer(x)

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
