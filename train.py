import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset/data.yaml', help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model to use')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--output', type=str, default='trained', help='Folder to save trained results')
    return parser.parse_args()


def main(opt):
    print(f'Training {opt.model} on {opt.data}')

    # Make sure output folder exists
    Path(opt.output).mkdir(parents=True, exist_ok=True)

    model = YOLO(opt.model)

    results = model.train(
        data=opt.data,
        epochs=opt.epochs,
        imgsz=opt.imgsz,
        batch=opt.batch,
        project=opt.output,
        name='.',
        device=0
    )

    # Move final best weights as yolov8n.pt inside output directory
    final_weight = Path(results.save_dir) / 'weights' / 'best.pt'
    target = Path(opt.output) / Path(opt.model).name
    final_weight.replace(target)
    print(f'\nSaved final model to: {target}')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
