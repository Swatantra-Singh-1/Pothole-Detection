import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run pothole detection on an image or video.")
    parser.add_argument("--source", type=str, required=True, help="Image, video, or folder path.")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/train/exp/weights/best.pt",
        help="Model weights path. If not available, YOLOv8 pre-trained weights are used.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference.")
    parser.add_argument("--project", type=str, default="runs/detect", help="Output project folder.")
    parser.add_argument("--name", type=str, default="pothole_detect", help="Output run name.")
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source)
    weights = Path(args.weights)

    if weights.exists():
        model = YOLO(str(weights))
    else:
        print(f"Weights not found at {weights}. Falling back to YOLOv8 pre-trained model.")
        model = YOLO("yolov8n.pt")

    results = model(
        str(source),
        conf=args.conf,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    print(f"Inference completed for: {source}")
    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes):
            print(f"Detected {len(boxes)} pothole object(s) with confidence threshold {args.conf}.")
            for i, box in enumerate(boxes):
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                print(f"Object {i + 1}: conf={conf:.3f}, bbox={xyxy}")
        else:
            print("No potholes detected in this input.")


if __name__ == "__main__":
    main()
