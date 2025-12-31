import numpy as np

class SyntheticDataGenerator:
    def __init__(self, seed=None, canvas_width=1000, canvas_height=1000):
        self.seed = seed
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        if seed is not None:
            np.random.seed(seed)

    def generate(self, count=10, mode='random', noise_level=0.0, **kwargs):
        """
        Generates a set of macros.
        Returns: (aligned_macros, disturbed_macros)
        """
        if mode == 'random':
            macros = self._generate_random(count, **kwargs)
            # Random has no "aligned" structure
            return macros, self._perturb_macros(macros, noise_level)
        
        elif mode == 'grid':
            aligned = self._generate_grid(count, **kwargs)
        elif mode == 'rows':
            aligned = self._generate_rows(count, **kwargs)
        elif mode == 'clustered':
             aligned = self._generate_clustered(count, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        disturbed = self._perturb_macros(aligned, noise_level)
        return aligned, disturbed

    def _generate_random(self, count, **kwargs):
        macros = []
        for i in range(count):
            w = 40 + np.random.rand() * 60
            h = 40 + np.random.rand() * 60
            x = np.random.rand() * (self.canvas_width - w)
            y = np.random.rand() * (self.canvas_height - h)
            macros.append({
                'id': i,
                'x': float(x),
                'y': float(y),
                'w': float(w),
                'h': float(h)
            })
        return macros

    def _generate_grid(self, count, grid_cols=None, **kwargs):
        if grid_cols is None:
            grid_cols = int(np.ceil(np.sqrt(count)))
        
        macros = []
        spacing = 20
        start_x, start_y = 50, 50
        cell_w, cell_h = 110, 110
        
        for i in range(count):
            w = 40 + np.random.rand() * 60
            h = 40 + np.random.rand() * 60
            
            row = i // grid_cols
            col = i % grid_cols
            
            cell_x = start_x + col * (cell_w + spacing)
            cell_y = start_y + row * (cell_h + spacing)
            
            x = cell_x + (cell_w - w) / 2
            y = cell_y + (cell_h - h) / 2
            
            macros.append({
                'id': i,
                'x': float(x),
                'y': float(y),
                'w': float(w),
                'h': float(h)
            })
        return macros

    def _generate_rows(self, count, rows=2, **kwargs):
        macros = []
        per_row = int(np.ceil(count / rows))
        spacing = 20
        start_x, start_y = 50, 50
        max_h_in_row = 100
        
        current_y = start_y
        for r in range(rows):
            current_x = start_x
            for c in range(per_row):
                idx = len(macros)
                if idx >= count: break
                
                w = 40 + np.random.rand() * 60
                h = 40 + np.random.rand() * 60
                
                macros.append({
                    'id': idx,
                    'x': float(current_x),
                    'y': float(current_y + (max_h_in_row - h) / 2),
                    'w': float(w),
                    'h': float(h)
                })
                current_x += w + spacing
            current_y += max_h_in_row + spacing + 20
        return macros

    def _generate_clustered(self, count, cluster_count=2, **kwargs):
        macros = []
        per_cluster = int(np.ceil(count / cluster_count))
        current_x, current_y = 50, 50
        max_h_in_row = 0
        spacing = 40
        cluster_spacing = 80
        
        for k in range(cluster_count):
            w = 40 + np.random.rand() * 60
            h = 40 + np.random.rand() * 60
            cluster_cols = int(np.ceil(np.sqrt(per_cluster)))
            cluster_rows = int(np.ceil(per_cluster / cluster_cols))
            
            cluster_width = cluster_cols * (w + spacing)
            if current_x + cluster_width > self.canvas_width:
                current_x = 50
                current_y += max_h_in_row + cluster_spacing
                max_h_in_row = 0
            
            cluster_height = cluster_rows * (h + spacing)
            max_h_in_row = max(max_h_in_row, cluster_height)
            
            for i in range(per_cluster):
                if len(macros) >= count: break
                row = i // cluster_cols
                col = i % cluster_cols
                x = current_x + col * (w + spacing)
                y = current_y + row * (h + spacing)
                macros.append({
                    'id': len(macros),
                    'x': float(x),
                    'y': float(y),
                    'w': float(w),
                    'h': float(h)
                })
            current_x += cluster_width + cluster_spacing
        return macros

    def _perturb_macros(self, macros, noise_level):
        if noise_level <= 0:
            return [m.copy() for m in macros]
        disturbed = []
        for m in macros:
            new_m = m.copy()
            dx = (np.random.rand() - 0.5) * 2 * noise_level
            dy = (np.random.rand() - 0.5) * 2 * noise_level
            new_m['x'] += dx
            new_m['y'] += dy
            new_m['x'] = np.clip(new_m['x'], 0, self.canvas_width - new_m['w'])
            new_m['y'] = np.clip(new_m['y'], 0, self.canvas_height - new_m['h'])
            disturbed.append(new_m)
        return disturbed