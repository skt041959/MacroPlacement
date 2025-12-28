import numpy as np

class SyntheticDataGenerator:
    def __init__(self, seed=None, canvas_width=1000, canvas_height=1000):
        self.seed = seed
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        if seed is not None:
            np.random.seed(seed)

    def generate(self, count=10, mode='random', **kwargs):
        if mode == 'random':
            return self._generate_random(count)
        elif mode == 'grid':
            return self._generate_grid(count, **kwargs)
        elif mode == 'rows':
            return self._generate_rows(count, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _generate_random(self, count):
        macros = []
        for i in range(count):
            w = 50 + np.random.rand() * 50
            h = 50 + np.random.rand() * 50
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

    def _generate_grid(self, count, grid_cols=None):
        if grid_cols is None:
            grid_cols = int(np.ceil(np.sqrt(count)))
        
        macros = []
        spacing = 20
        start_x, start_y = 50, 50
        
        # Fixed size for grid elements for simplicity in tests or predictable layout
        w, h = 60, 60
        
        for i in range(count):
            row = i // grid_cols
            col = i % grid_cols
            x = start_x + col * (w + spacing)
            y = start_y + row * (h + spacing)
            macros.append({
                'id': i,
                'x': float(x),
                'y': float(y),
                'w': float(w),
                'h': float(h)
            })
        return macros

    def _generate_rows(self, count, rows=2):
        macros = []
        per_row = int(np.ceil(count / rows))
        w, h = 60, 60
        spacing = 20
        
        for i in range(count):
            row_idx = i // per_row
            col_idx = i % per_row
            x = 50 + col_idx * (w + spacing)
            y = 50 + row_idx * (h + spacing + 50) # Extra vertical spacing for rows
            macros.append({
                'id': i,
                'x': float(x),
                'y': float(y),
                'w': float(w),
                'h': float(h)
            })
        return macros
