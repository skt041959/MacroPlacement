from dataset import RestorationDataset
from config import Config
import os

def generate_and_save():
    print(f"Starting dataset generation: {Config.NUM_TRAIN_SAMPLES} samples.")
    # This will trigger generation and saving because path is provided and doesn't exist yet
    dataset = RestorationDataset(
        num_samples=Config.NUM_TRAIN_SAMPLES, 
        seed=Config.SEED, 
        path=Config.DATASET_PATH
    )
    print(f"Dataset generation complete. Saved to {Config.DATASET_PATH}")

if __name__ == "__main__":
    generate_and_save()
