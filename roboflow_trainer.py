from pathlib import Path
from os.path import join
from roboflow import Roboflow
from rfdetr import RFDETRBase

VERSION = 2
API_KEY = "zxp6OFr5Vhj6Qa6JUjQi"
PROJECT_NAME = "cubicasa5k-2-qpmsa-gbbqv"


def get_paths():
    project_dir = Path(__file__).resolve().parent.resolve()
    dataset_dir = Path(join(project_dir, "dataset"))
    coco_dataset = Path(join(dataset_dir, "coco"))
    output_dir = Path(join(dataset_dir, "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    coco_dataset.mkdir(parents=True, exist_ok=True)
    return {
        "dir": project_dir,
        "output": output_dir,
        "coco": coco_dataset,
        "dataset": dataset_dir,
    }


def download_dataset(paths: dict):
    dataset_dir: Path = paths["coco"]
    is_filled = any(dataset_dir.iterdir())
    """Downloads dataset from Roboflow if dataset directory is empty."""
    if not is_filled:
        print("Dataset not found or empty. Downloading from Roboflow...")
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace().project(PROJECT_NAME)
        dataset = project.version(VERSION).download(
            "coco",
            location=str(dataset_dir),
            overwrite=True
        )
        print(f"Download complete. Dataset saved at: {dataset}")
    else:
        print(f"Dataset already exists at {dataset_dir}. Skipping download.")


def train_model(paths: dict):
    """Trains RF-DETR on the dataset."""
    dataset_dir: Path = paths["coco"]
    project_dir: Path = paths["dir"]
    # output_dir: Path = paths["output"]
    checkpoint = join(project_dir, "checkpoint.pth")
    print("train model dataset at:", dataset_dir)
    print("Initializing RF-DETR model...")
    model = RFDETRBase()
    history = []

    def callback2(data):
        history.append(data)

    model.callbacks["on_fit_epoch_end"].append(callback2)

    print("Starting training...")
    model.train(
        dataset_dir=str(dataset_dir),
        # output_dir=str(output_dir),
        epochs=50,
        batch_size=1,  # will fit your GPU
        grad_accum_steps=16,  # (1 * 16 = effective batch size of 16)
        lr=1e-4,
        resume=str(checkpoint)
    )

    # print(f"Training finished. Model checkpoints saved in: {output_dir}")


def main():
    paths = get_paths()
    download_dataset(paths)
    train_model(paths)


if __name__ == "__main__":
    main()
