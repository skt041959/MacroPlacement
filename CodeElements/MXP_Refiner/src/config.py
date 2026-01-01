# config.py
class Config:
    # 宏单元参数
    MACRO_COUNT = 100
    CANVAS_WIDTH = 1000.0
    CANVAS_HEIGHT = 1000.0
    
    # 构图阈值
    ALIGN_DIST_THRESH = 50.0   # 对齐边的最大距离
    ALIGN_SIZE_THRESH = 5.0    # 尺寸相似度容差
    PHYS_EDGE_CUTOFF = 1500.0   # Delaunay 边的修剪距离
    
    # 模型参数
    HIDDEN_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 4
    ACTION_DIM = 7  # [NoOp, Up, Down, Left, Right, Snap_X, Snap_Y]
    
    # RL 参数
    LR = 3e-4
    GAMMA = 0.99
    
    # 数据生成参数
    SEED = 42
    GENERATION_MODE = 'mixed' # 'random', 'grid', 'rows', 'clustered', 'mixed'
    GRID_COLS = 5
    NOISE_LEVEL = (5.0, 40.0) # Range for displacement
    NUM_TRAIN_SAMPLES = 10000
    DATASET_PATH = 'data/restoration_dataset_10k_mixed.pt'

    # 训练参数 (Restorer)
    RESTORER_EPOCHS = 100
    RESTORER_BATCH_SIZE = 32
    RESTORER_LR = 5e-4
    