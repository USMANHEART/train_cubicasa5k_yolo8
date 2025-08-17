import requests
import numpy as np
from io import BytesIO
from os.path import join
from pathlib import Path
from rfdetr import RFDETRBase
from PIL import Image, ImageDraw


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

def create_blank_image(size):
    """Create a blank white image of given size"""
    return Image.new('RGB', size, (255, 255, 255))

def draw_rectangles(image, detections, colors):
    """Draw rectangles on image with category-specific colors"""
    draw = ImageDraw.Draw(image)
    for det in detections:
        bbox = det['bbox']  # [x_min, y_min, x_max, y_max]
        label = det['label']
        color = colors[label]
        draw.rectangle(bbox, outline=color, width=3)
    return image


def save_results(paths, original_image, detections):
    # Create blank white image
    blank_image = create_blank_image(original_image.size)

    # Generate distinct colors for each category
    categories = set(det['label'] for det in detections)
    colors = {cat: tuple(np.random.randint(0, 256, size=3)) for cat in categories}

    # Draw rectangles on blank image
    result_image = draw_rectangles(blank_image, detections, colors)

    # Save result
    output_path = join(paths["output"], "result.png")
    result_image.save(output_path)
    print(f"Result saved to: {output_path}")


def parse_detections(results, class_names):
    """Convert model output to proper detection format"""
    detections = []

    for result in results:
        xyxy, _1, conf, class_id, _4, _5 = result
        label = class_names.get(int(class_id))
        detections.append({
            'bbox': xyxy.tolist(),
            'label': label,
            'score': float(conf)
        })

    return detections


def main():
    paths = get_paths()
    weights_path = paths["pretrain_weights"]
    model = RFDETRBase(pretrain_weights=str(weights_path))
    class_names = model.class_names

    url = "https://showlifecc-oss.eggrj.com/national_house/%E5%8C%97%E4%BA%AC%E5%B8%82-%E5%8C%97%E4%BA%AC%E5%B8%82-%E9%BE%99%E6%B9%BE%E5%88%AB%E5%A2%85-2%E5%AE%A40%E5%8E%852%E5%8D%AB-91.73%E3%8E%A1-4028bffa927a46590192a98180245a60-%E6%88%B7%E5%9E%8B%E5%9B%BE.jpg"
    image_file = Image.open(BytesIO(requests.get(url).content))
    results = model.predict(image_file, threshold=0.5)

    detections = parse_detections(results, class_names)
    print(detections)
    save_results(paths, image_file, detections)

if __name__ == "__main__":
    main()
