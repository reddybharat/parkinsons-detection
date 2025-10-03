import sys
import os

# Ensure the project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.train import Train

if __name__ == "__main__":
    print("[RUNNER] Starting model training...")
    trainer = Train()
    try:
        trainer.train()
        print("[RUNNER] Training completed successfully.")
    except Exception as e:
        print(f"[RUNNER] Training failed with exception: {e}")
