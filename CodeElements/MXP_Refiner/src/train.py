import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from config import Config
from model import HeteroGATRefiner
from env import MacroLayoutEnv
from utils import generate_random_data
from visualizer import DashboardGenerator

def train():
    # Hyperparameters
    num_episodes = 50
    max_steps_per_episode = 20
    learning_rate = Config.LR
    gamma = Config.GAMMA
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Metrics storage
    history = {
        'rewards': [],
        'losses': []
    }
    last_episode_snapshots = []

    # Initialize Environment and Model
    # ... (Initial generation just for model init dims)
    macros, netlist = generate_random_data()
    # env = MacroLayoutEnv(macros, netlist) # We re-init inside loop
    
    model = HeteroGATRefiner(
        hidden_dim=Config.HIDDEN_DIM, 
        out_dim=Config.ACTION_DIM,
        num_layers=Config.NUM_LAYERS, 
        num_heads=Config.NUM_HEADS
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    
    for episode in range(num_episodes):
        macros, netlist = generate_random_data()
        env = MacroLayoutEnv(macros, netlist)
        
        obs = env.get_graph_observation()
        total_reward = 0
        
        # Capture snapshots only for the last episode
        is_last_episode = (episode == num_episodes - 1)
        if is_last_episode:
            # Save deep copy of macros state
            last_episode_snapshots.append([m.copy() for m in env.macros])
        
        pbar = tqdm(range(max_steps_per_episode), desc=f"Episode {episode+1}/{num_episodes}")
        
        for step in pbar:
            # Prepare graph data
            obs = obs.to(device)
            
            # Construct edge_attr_dict
            edge_attr_dict = {
                ('macro', 'phys_edge', 'macro'): obs['macro', 'phys_edge', 'macro'].edge_attr,
                ('macro', 'logic_edge', 'macro'): obs['macro', 'logic_edge', 'macro'].edge_attr,
                ('macro', 'align_edge', 'macro'): None 
            }
            
            # Forward pass
            action_logits, value = model(obs.x_dict, obs.edge_index_dict, edge_attr_dict)
            
            # Action sampling
            probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            
            # Step environment
            next_obs, reward, done, _, _ = env.step(actions.cpu().numpy())
            next_obs = next_obs.to(device)
            
            if is_last_episode:
                last_episode_snapshots.append([m.copy() for m in env.macros])
            
            # Calculate next value for TD error
            next_edge_attr_dict = {
                ('macro', 'phys_edge', 'macro'): next_obs['macro', 'phys_edge', 'macro'].edge_attr,
                ('macro', 'logic_edge', 'macro'): next_obs['macro', 'logic_edge', 'macro'].edge_attr,
                ('macro', 'align_edge', 'macro'): None 
            }
            
            with torch.no_grad():
                _, next_value = model(next_obs.x_dict, next_obs.edge_index_dict, next_edge_attr_dict)
            
            # TD Target
            target_value = reward + gamma * next_value * (1 - int(done))
            advantage = target_value - value
            
            # Losses
            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = F.mse_loss(value, target_value)
            entropy = dist.entropy().mean()
            entropy_loss = -0.01 * entropy
            
            loss = actor_loss + critic_loss + entropy_loss
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_reward += reward
            history['losses'].append(loss.item())
            obs = next_obs
            
            pbar.set_postfix({'Reward': f"{reward:.2f}", 'Loss': f"{loss.item():.2f}"})
            
            if done:
                break
        
        history['rewards'].append(total_reward)
        print(f"Episode {episode+1} finished. Total Reward: {total_reward:.2f}")

    # Save model
    torch.save(model.state_dict(), "model.pth")
    print("Training finished. Model saved to model.pth")
    
    # Generate Dashboard
    print("Generating Dashboard...")
    viz = DashboardGenerator()
    # Pass the last used netlist
    viz.generate(history, last_episode_snapshots, netlist)

if __name__ == "__main__":
    train()
