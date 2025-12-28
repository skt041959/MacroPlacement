import torch
from config import Config
from model import HeteroGATRefiner
from env import MacroLayoutEnv
from utils import generate_random_data
import numpy as np

def main():
    # 1. 模拟数据初始化
    macros, netlist = generate_random_data()

    # 2. 初始化环境和模型
    env = MacroLayoutEnv(macros, netlist)
    model = HeteroGATRefiner(
        hidden_dim=Config.HIDDEN_DIM, 
        out_dim=Config.ACTION_DIM,
        num_layers=Config.NUM_LAYERS, 
        num_heads=Config.NUM_HEADS
    )
    
    # 3. 模拟推理循环
    obs = env.get_graph_observation()
    
    print(f"Graph Created: {obs}")
    print(f"Physical Edges: {obs['macro', 'phys_edge', 'macro'].edge_index.shape}")
    print(f"Align Edges: {obs['macro', 'align_edge', 'macro'].edge_index.shape}")
    
    # 4. 前向传播
    # 构造 edge_attr 字典
    edge_attr_dict = {
        ('macro', 'phys_edge', 'macro'): obs['macro', 'phys_edge', 'macro'].edge_attr,
        ('macro', 'logic_edge', 'macro'): obs['macro', 'logic_edge', 'macro'].edge_attr,
        # Align edge 暂时没有属性，设为 None (如果 GATv2Conv 不需要 edge_dim)
        # 或者在 DataBuilder 中添加 dummy attribute
        ('macro', 'align_edge', 'macro'): None 
    }
    
    action_logits, value = model(obs.x_dict, obs.edge_index_dict, edge_attr_dict)
    
    # 5. 动作采样
    probs = torch.softmax(action_logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    actions = dist.sample()
    
    print(f"Sampled Actions for {len(actions)} macros: {actions[:10]}...")
    
    # 6. 环境步进
    next_obs, reward, done, _, _ = env.step(actions.numpy())
    print(f"Step Reward: {reward:.4f}")

if __name__ == "__main__":
    import numpy as np
    main()