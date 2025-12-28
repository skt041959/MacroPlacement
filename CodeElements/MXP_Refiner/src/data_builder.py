import torch
import numpy as np
from scipy.spatial import Delaunay
from torch_geometric.data import HeteroData
from config import Config

class GraphBuilder:
    def __init__(self, macros, netlist):
        """
        macros: List of dict {'id': int, 'x': float, 'y': float, 'w': float, 'h': float}
        netlist: List of tuples (node_id_a, node_id_b, weight)
        """
        self.macros = macros
        self.netlist = netlist
        self.num_nodes = len(macros)

    def build_hetero_graph(self):
        data = HeteroData()
        
        # 1. 节点特征构建 [x_norm, y_norm, w_norm, h_norm]
        node_feats = []
        centers = []
        for m in self.macros:
            cx = m['x'] + m['w'] / 2
            cy = m['y'] + m['h'] / 2
            centers.append([cx, cy])
            # 归一化特征
            node_feats.append([
                cx / Config.CANVAS_WIDTH, 
                cy / Config.CANVAS_HEIGHT, 
                m['w'] / Config.CANVAS_WIDTH, 
                m['h'] / Config.CANVAS_HEIGHT
            ])
            
        data['macro'].x = torch.tensor(node_feats, dtype=torch.float)
        centers = np.array(centers)

        # 2. 构建物理边 (Physical Edges) - 基于 Delaunay
        # 作用：避障、防止重叠
        tri = Delaunay(centers)
        phys_edges = set()
        for simplex in tri.simplices:
            # simplex 是三角形的三个顶点索引
            for i in range(3):
                u, v = simplex[i], simplex[(i+1)%3]
                # 双向边
                phys_edges.add(tuple(sorted((u, v))))
        
        src_phys, dst_phys = [], []
        phys_attr = [] # 边特征：欧氏距离
        for u, v in phys_edges:
            dist = np.linalg.norm(centers[u] - centers[v])
            if dist < Config.PHYS_EDGE_CUTOFF:
                src_phys.extend([u, v])
                dst_phys.extend([v, u])
                # 归一化距离作为边特征
                dist_norm = dist / Config.CANVAS_WIDTH
                phys_attr.extend([[dist_norm], [dist_norm]])

        data['macro', 'phys_edge', 'macro'].edge_index = torch.tensor([src_phys, dst_phys], dtype=torch.long)
        data['macro', 'phys_edge', 'macro'].edge_attr = torch.tensor(phys_attr, dtype=torch.float)

        # 3. 构建对齐边 (Alignment Edges) - 基于尺寸匹配和投影重叠
        # 作用：长程吸附，形成阵列
        src_align, dst_align = [], []
        # 简化版扫描：这里用 O(N^2) 演示逻辑，生产环境应用区间树(Interval Tree)优化
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                mi, mj = self.macros[i], self.macros[j]
                
                # 检查尺寸是否相似 (Width or Height match)
                w_sim = abs(mi['w'] - mj['w']) < Config.ALIGN_SIZE_THRESH
                h_sim = abs(mi['h'] - mj['h']) < Config.ALIGN_SIZE_THRESH
                
                # 检查是否处于“对齐通道”内 (X 或 Y 轴投影接近)
                x_align = abs(centers[i][0] - centers[j][0]) < Config.ALIGN_DIST_THRESH
                y_align = abs(centers[i][1] - centers[j][1]) < Config.ALIGN_DIST_THRESH
                
                # 逻辑：如果尺寸相似且在同一轴线上，建立对齐边
                if (w_sim and x_align) or (h_sim and y_align):
                    src_align.extend([i, j])
                    dst_align.extend([j, i])

        data['macro', 'align_edge', 'macro'].edge_index = torch.tensor([src_align, dst_align], dtype=torch.long)

        # 4. 构建逻辑边 (Logic Edges) - 基于网表
        # 作用：线长优化
        src_logic, dst_logic = [], []
        logic_weights = []
        for u, v, w in self.netlist:
            src_logic.extend([u, v])
            dst_logic.extend([v, u])
            logic_weights.extend([[w], [w]]) # 边特征：连接权重
            
        data['macro', 'logic_edge', 'macro'].edge_index = torch.tensor([src_logic, dst_logic], dtype=torch.long)
        data['macro', 'logic_edge', 'macro'].edge_attr = torch.tensor(logic_weights, dtype=torch.float)

        return data