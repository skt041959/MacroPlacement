import torch
import numpy as np
from tqdm import tqdm
from config import Config
from model import GraphUNet
from diffusion import DiffusionScheduler
from geometry import alignment_energy_function
from data_builder import GraphBuilder

class FloorplanRestorationInference:
    def __init__(self, model_path="restorer_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize GraphUNet
        self.model = GraphUNet(
            hidden_dim=Config.HIDDEN_DIM, 
            num_layers=Config.NUM_LAYERS, 
            num_heads=Config.NUM_HEADS
        ).to(self.device)
        
        self.diffusion = DiffusionScheduler()
        # Move scheduler tensors to device
        self.diffusion.sqrt_alphas_cumprod = self.diffusion.sqrt_alphas_cumprod.to(self.device)
        self.diffusion.sqrt_one_minus_alphas_cumprod = self.diffusion.sqrt_one_minus_alphas_cumprod.to(self.device)
        self.diffusion.betas = self.diffusion.betas.to(self.device)
        self.diffusion.alphas = self.diffusion.alphas.to(self.device)
        self.diffusion.alphas_cumprod = self.diffusion.alphas_cumprod.to(self.device)
        self.diffusion.alphas_cumprod_prev = self.diffusion.alphas_cumprod_prev.to(self.device)
        
        try:
            # key mismatch is expected during architecture development
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True), strict=False)
            print(f"Loaded model from {model_path} (strict=False)")
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            print("Using uninitialized model.")
            
        self.model.eval()

    def restore(self, disturbed_macros, guidance_scale=None):
        """
        disturbed_macros: list of dicts {'id', 'x', 'y', 'w', 'h', ...}
        guidance_scale: float, strength of alignment guidance (default: from Config)
        Returns: restored_macros (same format)
        """
        guidance_scale = guidance_scale if guidance_scale is not None else Config.GUIDANCE_SCALE
        
        # 1. Build initial graph structure
        # Note: We need the graph topology (edges), but the initial 'x' (node features) 
        # will only provide 'w', 'h' (types). The 'x', 'y' coords will be initialized as noise.
        builder = GraphBuilder(disturbed_macros, netlist=[])
        data = builder.build_hetero_graph().to(self.device)
        
        num_nodes = data.x_dict['macro'].size(0)
        
        # 2. Initialize pure Gaussian noise x_T
        x = torch.randn((num_nodes, 2), device=self.device)
        
        # Static edge attributes
        edge_attr_dict = {
            ('macro', 'phys_edge', 'macro'): data['macro', 'phys_edge', 'macro'].edge_attr,
            ('macro', 'logic_edge', 'macro'): data['macro', 'logic_edge', 'macro'].edge_attr,
            ('macro', 'align_edge', 'macro'): None 
        }
        
        # 3. DDPM Sampling Loop (Reverse Process)
        # Iterate from T-1 down to 0
        timesteps = range(self.diffusion.num_steps - 1, -1, -1)
        
        with torch.no_grad(): # Mostly no grad, except for guidance
             for t_idx in tqdm(timesteps, desc="Restoring Layout"):
                t = torch.tensor([t_idx], device=self.device).repeat(num_nodes)
                
                # Setup requires_grad for Guidance
                x_in = x.detach().clone()
                x_in.requires_grad = True
                
                # Construct input dictionary
                # Features: [x, y, w, h] -> we replace x,y with current noisy state
                current_feats = torch.cat([x_in, data.x_dict['macro'][:, 2:]], dim=-1)
                x_dict = {'macro': current_feats}
                
                # Predict Noise epsilon_theta
                # Note: We briefly enable grad here ONLY to backprop through the Energy Function
                # onto x_in, NOT to update model weights. But standard DDPM guidance is usually:
                # epsilon_modified = epsilon_theta - scale * grad(log p(y|x))
                # Here we use: grad(Energy)
                
                # A. Model Prediction (Gradient-free for weights)
                with torch.set_grad_enabled(False):
                    pred_noise = self.model(x_dict, data.edge_index_dict, edge_attr_dict, t)
                
                # B. Alignment Guidance (Gradient required wrt x_in)
                if guidance_scale > 0:
                    with torch.enable_grad():
                        energy = alignment_energy_function(
                            x_in, 
                            data.edge_index_dict.get(('macro', 'align_edge', 'macro')),
                            grid_pitch=Config.GRID_K / Config.CANVAS_WIDTH, # Normalized grid
                            lambda_grid=Config.LAMBDA_GRID,
                            lambda_channel=Config.LAMBDA_CHANNEL
                        )
                        # Compute gradient: d(Energy)/dx_t
                        grad = torch.autograd.grad(energy, x_in)[0]
                        
                    # Modify noise: epsilon' = epsilon + s * grad
                    # If Energy is "negative log probability", then grad is analogous to classifier guidance
                    # We want to minimize Energy -> move x opposed to grad(Energy)
                    # But DDPM formulation usually subtracts predicted noise.
                    # x_{t-1} = ... - sigma * epsilon
                    # So adding efficient guidance means modifying epsilon to point TOWARDS high energy?
                    # No, we want low energy. 
                    # Standard classifier guidance: eps_hat = eps - w * grad(log p(y|x_t))
                    # Here -Energy ~ log p(alignment). 
                    # So grad(log p) ~ -grad(Energy).
                    # eps_hat = eps + w * grad(Energy)
                    
                    pred_noise = pred_noise + guidance_scale * grad
                
                # C. Compute x_{t-1}
                beta_t = self.diffusion.betas[t_idx]
                alpha_t = self.diffusion.alphas[t_idx]
                alpha_cumprod_t = self.diffusion.alphas_cumprod[t_idx]
                alpha_cumprod_prev_t = self.diffusion.alphas_cumprod_prev[t_idx]
                sqrt_one_minus_alpha_cumprod_t = self.diffusion.sqrt_one_minus_alphas_cumprod[t_idx]
                
                # Standard DDPM Update:
                # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (beta_t / sqrt(1-alpha_bar)) * eps) + sigma * z
                
                coef1 = 1 / torch.sqrt(alpha_t)
                coef2 = beta_t / sqrt_one_minus_alpha_cumprod_t
                
                mean = coef1 * (x - coef2 * pred_noise)
                
                if t_idx > 0:
                    noise = torch.randn_like(x)
                    sigma_t = torch.sqrt(beta_t) # Option 1: fixed sigma
                    x = mean + sigma_t * noise
                else:
                    x = mean # No noise at last step

        # 4. Denormalize coordinates
        pred_coords = x.detach().cpu().numpy() # [N, 2]
        
        restored_macros = []
        for i, m in enumerate(disturbed_macros):
            new_m = m.copy()
            # pred_coords are centers normalized
            cx = pred_coords[i, 0] * Config.CANVAS_WIDTH
            cy = pred_coords[i, 1] * Config.CANVAS_HEIGHT
            
            # Convert back to bottom-left (x, y)
            new_m['x'] = float(cx - m['w'] / 2)
            new_m['y'] = float(cy - m['h'] / 2)
            
            # Clip to canvas
            new_m['x'] = np.clip(new_m['x'], 0, Config.CANVAS_WIDTH - new_m['w'])
            new_m['y'] = np.clip(new_m['y'], 0, Config.CANVAS_HEIGHT - new_m['h'])
            
            restored_macros.append(new_m)
            
        return restored_macros

if __name__ == "__main__":
    from generator import SyntheticDataGenerator
    
    # Simple test
    gen = SyntheticDataGenerator(seed=42)
    aligned, disturbed = gen.generate(count=20, mode='clustered', noise_level=Config.NOISE_LEVEL)
    
    inference = FloorplanRestorationInference()
    print("Starting Diffusion Restoration...")
    restored = inference.restore(disturbed, guidance_scale=Config.GUIDANCE_SCALE)
    
    print(f"Restored {len(restored)} macros.")
    print(f"First macro Aligned: ({aligned[0]['x']:.2f}, {aligned[0]['y']:.2f})")
    print(f"First macro Disturbed: ({disturbed[0]['x']:.2f}, {disturbed[0]['y']:.2f})")
    print(f"First macro Restored: ({restored[0]['x']:.2f}, {restored[0]['y']:.2f})")
