import requests
from PIL import Image
from io import BytesIO
from os.path import join
from pathlib import Path
from rfdetr import RFDETRBase


def get_paths():
    project_dir = Path(__file__).resolve().parent.resolve()
    pretrain_weights = Path(join(project_dir, "weights.pt"))
    output_dir = Path(join(project_dir, "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dir": project_dir,
        "output": output_dir,
        "pretrain_weights": pretrain_weights,
    }


def main():
    paths = get_paths()
    weights_path = paths["pretrain_weights"]
    url = "https://showlifecc-oss.eggrj.com/national_house/北京市-北京市-四合院-7室5厅3卫-387.54㎡-ff808081965b8a890196b9bf5f8c0b9c-户型图.jpg"
    image_file = Image.open(BytesIO(requests.get(url).content))

    model = RFDETRBase(pretrain_weights=weights_path)

    # Run prediction
    results = model.predict(image_file, threshold=0.5)
    print("Prediction results:")
    print(results)

if __name__ == "__main__":
    main()
