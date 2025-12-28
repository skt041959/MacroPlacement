import gymnasium as gym
import numpy as np
from gymnasium import spaces
from data_builder import GraphBuilder
from config import Config

class MacroLayoutEnv(gym.Env):
    def __init__(self, initial_macros, netlist):
        super().__init__()
        self.macros = initial_macros # 深拷贝
        self.netlist = netlist
        
        # 动作空间：每个宏独立决策 (Batch Concurrent)
        # 0: NoOp
        # 1-4: Shift (Up, Down, Left, Right) 1 unit
        # 5: Snap to nearest X grid/neighbor
        # 6: Snap to nearest Y grid/neighbor
        self.action_space = spaces.Discrete(Config.ACTION_DIM)
        
        # 观察空间：这里简化为占位，实际是 HeteroData
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.macros), 4))

    def get_graph_observation(self):
        builder = GraphBuilder(self.macros, self.netlist)
        return builder.build_hetero_graph()

    def step(self, actions):
        """
        actions: List or Tensor of shape [Num_Macros], containing action indices
        """
        prev_metrics = self._calculate_metrics()
        
        # 1. 执行动作 (并发更新坐标)
        for i, action in enumerate(actions):
            self._apply_action(i, action)
            
        # 2. 重新构图 (动态图更新)
        # 在工程优化中，Delaunay不必每步全量更新，可局部修复
        obs = self.get_graph_observation()
        
        # 3. 计算奖励
        curr_metrics = self._calculate_metrics()
        reward = self._compute_reward(prev_metrics, curr_metrics)
        
        # 4. 检查终止条件 (例如达到步数上限或 DRC Clean)
        done = False 
        
        return obs, reward, done, False, {}

    def _apply_action(self, idx, action):
        macro = self.macros[idx]
        step_size = 5.0
        
        if action == 1: macro['y'] += step_size
        elif action == 2: macro['y'] -= step_size
        elif action == 3: macro['x'] -= step_size
        elif action == 4: macro['x'] += step_size
        elif action == 5: self._snap_x(idx) # 对齐逻辑
        elif action == 6: self._snap_y(idx) # 对齐逻辑
        
        # 边界限制
        macro['x'] = np.clip(macro['x'], 0, Config.CANVAS_WIDTH - macro['w'])
        macro['y'] = np.clip(macro['y'], 0, Config.CANVAS_HEIGHT - macro['h'])

    def _snap_x(self, idx):
        # Placeholder: snap to nearest 10 units
        self.macros[idx]['x'] = round(self.macros[idx]['x'] / 10.0) * 10.0

    def _snap_y(self, idx):
        # Placeholder: snap to nearest 10 units
        self.macros[idx]['y'] = round(self.macros[idx]['y'] / 10.0) * 10.0

    def _calculate_metrics(self):
        # 计算 HPWL, 重叠面积, 对齐分数
        # 此处省略具体几何计算代码
        return {'hpwl': 1000, 'overlap': 50, 'alignment': 10}

    def _compute_reward(self, prev, curr):
        # 复合奖励函数
        r_hpwl = (prev['hpwl'] - curr['hpwl']) * 0.5
        r_overlap = (prev['overlap'] - curr['overlap']) * 2.0 # 重惩罚重叠
        r_align = (curr['alignment'] - prev['alignment']) * 1.0 # 奖励对齐
        return r_hpwl + r_overlap + r_align