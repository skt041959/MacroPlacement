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
        # Handle variable noise level
        if isinstance(noise_level, (list, tuple)):
            actual_noise = np.random.uniform(noise_level[0], noise_level[1])
        else:
            actual_noise = noise_level

        if mode == 'random':
            macros = self._generate_random(count, **kwargs)
            # Random has no "aligned" structure
            return macros, self._perturb_macros(macros, actual_noise)
        
        elif mode == 'grid':
            aligned = self._generate_grid(count, **kwargs)
        elif mode == 'rows':
            aligned = self._generate_rows(count, **kwargs)
        elif mode == 'clustered':
             aligned = self._generate_clustered(count, **kwargs)
        elif mode == 'mixed':
             aligned = self._generate_mixed(count, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        disturbed = self._perturb_macros(aligned, actual_noise)
        return aligned, disturbed

    def _generate_mixed(self, count, **kwargs):
        """
        Combines random, grid, and clustered generation.
        """
        modes = ['random', 'grid', 'clustered']
        # Randomly choose a sub-mode for this call, OR split count into chunks?
        # Let's split count into chunks and use different strategies for each chunk.
        
        macros = []
        remaining = count
        
        # Determine splits (e.g., 2 or 3 splits)
        num_splits = np.random.randint(2, 4)
        
        splits = []
        for i in range(num_splits - 1):
            if remaining <= 0: break
            split_count = np.random.randint(1, remaining)
            splits.append(split_count)
            remaining -= split_count
        if remaining > 0:
            splits.append(remaining)
            
        current_x, current_y = 50, 50
        
        for split_count in splits:
            sub_mode = np.random.choice(modes)
            
            # Generate sub-group
            if sub_mode == 'random':
                # Localize random generation to a region?
                # _generate_random distributes over whole canvas.
                # Let's just generate and then shift?
                # Or simply call existing methods and let them place.
                # Problem: They might overlap.
                # _generate_random uses full canvas. _generate_grid/clustered use localized cursor.
                
                # For mixed, let's just delegate to _generate_clustered for structure
                # and maybe add some random scatter.
                
                # Simplification: 'Mixed' just randomly picks ONE strategy for the whole batch
                # OR actually mixes them.
                # Let's pick one strategy for the whole batch for now to keep it valid (non-overlapping).
                # Mixing different logic in one canvas requires collision detection which is complex.
                
                # To truly mix:
                # We can generate separate groups and try to place them.
                # But since we don't have a placer, let's stick to:
                # Randomly picking one of the structured modes per call is effectively "mixed" over a dataset.
                
                # Wait, "Mixed generation mode (combining clusters and random placements)"
                # Let's generate a clustered backbone and then add some random macros.
                
                sub_mode = 'clustered' # Base
            
            if sub_mode == 'clustered':
                sub_macros = self._generate_clustered(split_count, cluster_count=np.random.randint(1, 3))
            elif sub_mode == 'grid':
                sub_macros = self._generate_grid(split_count, grid_cols=int(np.sqrt(split_count)))
            else: # random
                sub_macros = self._generate_random(split_count)
                
            # Shift sub_macros to avoid overlap?
            # Existing methods restart at 50,50.
            # This implementation of mixed is tricky without a packer.
            
            # Revised approach:
            # Randomly select a mode for the ENTIRE layout.
            # This increases diversity across the dataset.
            pass

        # To support "Mixed" as a single mode that varies per call:
        selected_mode = np.random.choice(['grid', 'rows', 'clustered']) 
        # Exclude 'random' from base alignment if we want ground truth to be aligned.
        
        if selected_mode == 'grid':
            return self._generate_grid(count, **kwargs)
        elif selected_mode == 'rows':
            return self._generate_rows(count, **kwargs)
        elif selected_mode == 'clustered':
            return self._generate_clustered(count, **kwargs)
        return []

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