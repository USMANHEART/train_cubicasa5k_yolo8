import subprocess
from pathlib import Path
from roboflow import Roboflow


API_KEY = "zxp6OFr5Vhj6Qa6JUjQi"
PROJECT_NAME = "cubicasa5k-2-qpmsa-gbbqv"
VERSION = "2"
DATASET_DIR = Path("dataset/coco")

TRAIN_SCRIPT = "detr/train.py"
EPOCHS = 100
BATCH_SIZE = 4
NUM_CLASSES = 5
DEVICE = "cuda"  # or 'cpu'


def download_dataset():
    """Downloads dataset from Roboflow if dataset directory is empty."""
    if not DATASET_DIR.exists() or not any(DATASET_DIR.iterdir()):
        print("Dataset not found or empty. Downloading from Roboflow...")
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace().project(PROJECT_NAME)
        dataset = project.version(VERSION).download("coco")
        print("Download complete.")
    else:
        print(f"Dataset already exists at {DATASET_DIR}. Skipping download.")


def start_training():
    """Launches DETR training using subprocess."""
    cmd = [
        "python", TRAIN_SCRIPT,
        "--dataset", str(DATASET_DIR),
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--num-classes", str(NUM_CLASSES),
        "--device", DEVICE
    ]
    print("Starting training...")
    subprocess.run(cmd)


def main():
    download_dataset()
    start_training()


if __name__ == "__main__":
    main()
