from dataset import RestorationDataset
from config import Config
import torch
import os

def generate_categorized_datasets():
    print(f"Starting categorized dataset generation. Total target: {Config.NUM_TRAIN_SAMPLES} per category.")
    
    combined_data = []
    
    for category in Config.CATEGORIES:
        path = Config.DATASET_PATH_TEMPLATE.format(category)
        print(f"\n--- Generating Category: {category} ---")
        
        # If file exists, we could skip, but let's overwrite to ensure fresh logic
        if os.path.exists(path):
            os.remove(path)
            
        dataset = RestorationDataset(
            num_samples=Config.NUM_TRAIN_SAMPLES, 
            mode=category,
            seed=Config.SEED, 
            path=path
        )
        print(f"Category {category} complete. Saved to {path}")
        combined_data.extend(dataset.data_list)
        
    # Also save a combined version for convenience
    print(f"\nSaving combined dataset to {Config.DATASET_PATH}")
    torch.save(combined_data, Config.DATASET_PATH)
    
    # 2. Perform Train/Val Split
    print("\n--- Performing Train/Val Split ---")
    import random
    random.seed(Config.SEED)
    random.shuffle(combined_data)
    
    num_val = int(len(combined_data) * Config.VAL_RATIO)
    val_data = combined_data[:num_val]
    train_data = combined_data[num_val:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    
    torch.save(train_data, Config.TRAIN_DATA_PATH)
    torch.save(val_data, Config.VAL_DATA_PATH)
    
    print(f"Datasets saved to {Config.TRAIN_DATA_PATH} and {Config.VAL_DATA_PATH}")
    print("All operations complete.")

if __name__ == "__main__":
    generate_categorized_datasets()
