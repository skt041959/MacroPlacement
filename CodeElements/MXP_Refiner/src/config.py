# config.py
class Config:
    # 宏单元参数
    MACRO_COUNT = 100
    CANVAS_WIDTH = 1000.0
    CANVAS_HEIGHT = 1000.0
    
    # 构图阈值
    ALIGN_DIST_THRESH = 50.0   # 对齐边的最大距离
    ALIGN_SIZE_THRESH = 5.0    # 尺寸相似度容差
    PHYS_EDGE_CUTOFF = 300.0   # Delaunay 边的修剪距离
    
    # 模型参数
    HIDDEN_DIM = 64
    NUM_HEADS = 4
    NUM_LAYERS = 3
    ACTION_DIM = 7  # [NoOp, Up, Down, Left, Right, Snap_X, Snap_Y]
    
    # RL 参数
    LR = 3e-4
    GAMMA = 0.99
    
    # 数据生成参数
    SEED = 42
    GENERATION_MODE = 'random' # 'random', 'grid', 'rows'
    GRID_COLS = 10