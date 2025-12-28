import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear

class HeteroGATRefiner(nn.Module):
    def __init__(self, hidden_dim, out_dim, num_layers, num_heads):
        super().__init__()
        
        # 1. 初始特征编码
        # 将 4维节点特征 (x,y,w,h) 投影到隐藏层
        self.node_encoder = Linear(4, hidden_dim)
        
        # 2. 异构图卷积层
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                # 物理边：关注距离，提取避障信息
                ('macro', 'phys_edge', 'macro'): GATv2Conv(
                    hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                    edge_dim=1, add_self_loops=False
                ),
                # 对齐边：关注几何关系，提取对齐信息
                ('macro', 'align_edge', 'macro'): GATv2Conv(
                    hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                    add_self_loops=False
                ),
                # 逻辑边：关注连接权重，提取拉力信息
                ('macro', 'logic_edge', 'macro'): GATv2Conv(
                    hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                    edge_dim=1, add_self_loops=False
                ),
            }, aggr='sum') # 将三种边的信息求和聚合
            self.convs.append(conv)

        # 3. 策略头 (Actor) - 输出每个宏的动作概率
        self.actor_head = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, out_dim)
        )
        
        # 4. 价值头 (Critic) - 评估当前布局好坏
        self.critic_head = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, 1)
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # x_dict['macro']: [N, 4]
        x = self.node_encoder(x_dict['macro'])
        x_dict['macro'] = x

        for conv in self.convs:
            # HeteroConv 需要传递 edge_attr 字典
            # 注意：PyG 的 HeteroConv 调用方式可能随版本略有变化，此处为通用逻辑
            out_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: F.elu(x) for key, x in out_dict.items()}
        
        # 最终节点嵌入
        node_embed = x_dict['macro'] # [N, Hidden]
        
        # 全局池化 (Readout) 用于 Critic
        global_embed = torch.mean(node_embed, dim=0) 
        
        # 策略输出 logits
        action_logits = self.actor_head(node_embed) # [N, Action_Dim]
        value = self.critic_head(global_embed)      # [1]
        
        return action_logits, value