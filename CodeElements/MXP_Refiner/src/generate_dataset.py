from dataset import RestorationDataset
from config import Config
import torch
import os

def generate_categorized_datasets():
    print(f"Starting categorized dataset generation.")
    
    # Define Split Sizes
    # Total was 5000 per category. Let's keep that ratio.
    train_samples_per_cat = int(Config.NUM_TRAIN_SAMPLES * (1 - Config.VAL_RATIO))
    val_samples_per_cat = int(Config.NUM_TRAIN_SAMPLES * Config.VAL_RATIO)
    
    train_data = []
    val_data = []
    
    for category in Config.CATEGORIES:
        print(f"\n--- Processing Category: {category} ---")
        
        # 1. Generate Training Data (Reference Topology)
        print(f"Generating Training Data for {category} (Ref Topology)...")
        train_ds = RestorationDataset(
            num_samples=train_samples_per_cat, 
            mode=category,
            seed=Config.SEED, 
            path=None, # Don't save intermediate per-category files to avoid clutter/confusion
            use_reference_topology=True
        )
        train_data.extend(train_ds.data_list)
        
        # 2. Generate Validation Data (Disturbed Topology)
        print(f"Generating Validation Data for {category} (Disturbed Topology)...")
        # Use a different seed offset for validation to ensure no leakage
        val_ds = RestorationDataset(
            num_samples=val_samples_per_cat, 
            mode=category,
            seed=Config.SEED + 999, 
            path=None,
            use_reference_topology=False
        )
        val_data.extend(val_ds.data_list)
        
    # Shuffle
    import random
    random.seed(Config.SEED)
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    print(f"\nTotal Train samples: {len(train_data)}")
    print(f"Total Val samples: {len(val_data)}")
    
    torch.save(train_data, Config.TRAIN_DATA_PATH)
    torch.save(val_data, Config.VAL_DATA_PATH)
    
    print(f"Datasets saved to {Config.TRAIN_DATA_PATH} and {Config.VAL_DATA_PATH}")
    print("All operations complete.")

if __name__ == "__main__":
    generate_categorized_datasets()
