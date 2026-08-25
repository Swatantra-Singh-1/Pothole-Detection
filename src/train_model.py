import argparse
from pathlib import Path

from ultralytics import YOLO

from config import DEFAULT_DATASET, DEFAULT_MODEL


def parse_args():
    parser = argparse.ArgumentParser(description="Train a pothole detection model with YOLOv8.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATASET), help="Dataset YAML path.")
    parser.add_argument("--weights", type=str, default=DEFAULT_MODEL, help="Base weights to start from.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--project", type=str, default="runs/train", help="Project directory for outputs.")
    parser.add_argument("--name", type=str, default="exp", help="Run name.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.data)

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Create a YOLO-style dataset YAML and point --data to it.")
        print("Example dataset structure:")
        print("data/")
        print("  images/")
        print("    train/")
        print("    val/")
        print("  labels/")
        print("    train/")
        print("    val/")
        print("  pothole_dataset.yaml")
        return

    model = YOLO(args.weights)
    model.train(
        data=str(dataset_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    print("Training complete.")
    save_dir = getattr(model.trainer, "save_dir", Path(args.project) / args.name)
    print(f"Best weights saved in: {Path(save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
