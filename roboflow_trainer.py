from pathlib import Path
from roboflow import Roboflow
from rfdetr import RFDETRBase

# 1️⃣ Configuration
API_KEY = "zxp6OFr5Vhj6Qa6JUjQi"
PROJECT_NAME = "cubicasa5k-2-qpmsa-gbbqv"
VERSION = "2"
DATASET_DIR = Path("dataset/coco")
OUTPUT_DIR = Path("trained_model")

EPOCHS = 50
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LR = 1e-4
DEVICE = "cuda"  # Change to "cpu" if no GPU
EARLY_STOPPING = True


def download_dataset():
    """Downloads dataset from Roboflow if dataset directory is empty."""
    if not DATASET_DIR.exists() or not any(DATASET_DIR.iterdir()):
        print("Dataset not found or empty. Downloading from Roboflow...")
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace().project(PROJECT_NAME)
        dataset = project.version(VERSION).download("coco")
        print(f"Download complete. Dataset saved at: {DATASET_DIR}")
    else:
        print(f"Dataset already exists at {DATASET_DIR}. Skipping download.")


def train_model():
    """Trains RF-DETR on the dataset."""
    print("Initializing RF-DETR model...")
    model = RFDETRBase()

    print("Starting training...")
    model.train(
        dataset_dir=str(DATASET_DIR),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        lr=LR,
        output_dir=str(OUTPUT_DIR),
        device=DEVICE,
        early_stopping=EARLY_STOPPING
    )
    print(f"Training finished. Model checkpoints saved in: {OUTPUT_DIR}")


def main():
    download_dataset()
    train_model()


if __name__ == "__main__":
    main()
