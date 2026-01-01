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
                'id': i, 'x': float(x), 'y': float(y), 'w': float(w), 'h': float(h)
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
                    'id': idx, 'x': float(current_x), 'y': float(current_y + (max_h_in_row - h) / 2),
                    'w': float(w), 'h': float(h)
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
                    'id': len(macros), 'x': float(x), 'y': float(y), 'w': float(w), 'h': float(h)
                })
            current_x += cluster_width + cluster_spacing
        return macros

    def _generate_mixed(self, count, **kwargs):
        """
        Clarified Mixed Mode:
        - Divide 'count' macros into multiple clusters (Dynamic Count).
        - Use a small pool of 2 distinct macro sizes.
        - Each cluster is aligned in a variable grid shape.
        - All macros are structured within clusters.
        """
        num_clusters = np.random.randint(2, 5)
        
        # 1. Split count into clusters
        cluster_counts = []
        remaining = count
        for i in range(num_clusters - 1):
            if remaining <= 1: break
            # Ensure at least 1 macro per cluster, but try to keep them decent sized
            upper = max(2, remaining // 2)
            c = np.random.randint(1, upper + 1)
            cluster_counts.append(c)
            remaining -= c
        if remaining > 0:
            cluster_counts.append(remaining)
        
        # 2. Define two distinct macro sizes for the pool
        size_pool = [
            (40 + np.random.rand() * 20, 40 + np.random.rand() * 20), # Size Type 1 (Small)
            (70 + np.random.rand() * 30, 70 + np.random.rand() * 30)  # Size Type 2 (Large)
        ]
        
        macros = []
        current_x, current_y = 50, 50
        max_h_in_row = 0
        spacing = 15
        cluster_spacing = 50
        
        for c_count in cluster_counts:
            # Pick a size from pool
            w, h = size_pool[np.random.randint(0, len(size_pool))]
            
            # 3. Determine grid shape (randomly biased towards square-ish)
            ideal_cols = int(np.ceil(np.sqrt(c_count)))
            # Add some variation to the shape
            cols = np.random.randint(max(1, ideal_cols - 1), ideal_cols + 2)
            rows = int(np.ceil(c_count / cols))
            
            cluster_w = cols * w + (cols - 1) * spacing
            cluster_h = rows * h + (rows - 1) * spacing
            
            # 4. Simple row-based packing for clusters on the canvas
            if current_x + cluster_w > self.canvas_width - 50:
                current_x = 50
                current_y += max_h_in_row + cluster_spacing
                max_h_in_row = 0
            
            # Stop if we run out of vertical space
            if current_y + cluster_h > self.canvas_height - 50:
                break
                
            # Place macros in this cluster
            for i in range(c_count):
                r = i // cols
                c = i % cols
                
                macros.append({
                    'id': len(macros),
                    'x': float(current_x + c * (w + spacing)),
                    'y': float(current_y + r * (h + spacing)),
                    'w': float(w),
                    'h': float(h)
                })
            
            max_h_in_row = max(max_h_in_row, cluster_h)
            current_x += cluster_w + cluster_spacing
            
        return macros

    def _perturb_macros(self, macros, noise_level):
        # Handle range
        if isinstance(noise_level, (list, tuple)):
            actual_noise = np.random.uniform(noise_level[0], noise_level[1])
        else:
            actual_noise = noise_level

        if actual_noise <= 0:
            return [m.copy() for m in macros]
            
        disturbed = []
        for m in macros:
            new_m = m.copy()
            dx = (np.random.rand() - 0.5) * 2 * actual_noise
            dy = (np.random.rand() - 0.5) * 2 * actual_noise
            new_m['x'] += dx
            new_m['y'] += dy
            new_m['x'] = np.clip(new_m['x'], 0, self.canvas_width - new_m['w'])
            new_m['y'] = np.clip(new_m['y'], 0, self.canvas_height - new_m['h'])
            disturbed.append(new_m)
        return disturbed
