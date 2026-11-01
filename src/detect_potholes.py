import argparse
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

from ultralytics import YOLO

from geotag_detection import DetectionStore, NmeaGpsReader


def parse_args():
    parser = argparse.ArgumentParser(description="Run pothole detection on an image or video.")
    parser.add_argument("--source", type=str, required=True, help="Image, video, folder path, or camera index such as 0.")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/train/exp/weights/best.pt",
        help="Model weights path. If not available, YOLOv8 pre-trained weights are used.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference.")
    parser.add_argument("--show", action="store_true", help="Display the live annotated video while processing.")
    parser.add_argument("--project", type=str, default="runs/detect", help="Output project folder.")
    parser.add_argument("--name", type=str, default="pothole_detect", help="Output run name.")
    parser.add_argument("--gps-port", type=str, help="GPS receiver serial port, for example COM3.")
    parser.add_argument("--gps-baud", type=int, default=9600, help="GPS receiver baud rate.")
    parser.add_argument("--gps-max-age", type=float, default=5.0, help="Maximum age in seconds for a GPS fix.")
    parser.add_argument(
        "--database",
        type=str,
        default="data/pothole_detections.sqlite3",
        help="SQLite database used when --gps-port is set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source
    weights = Path(args.weights)

    if weights.exists():
        model = YOLO(str(weights))
    else:
        print(f"Weights not found at {weights}. Falling back to YOLOv8 pre-trained model.")
        model = YOLO("yolov8n.pt")

    run_id = uuid4().hex
    saved_track_ids = set()
    gps_warning_shown = False
    tracking_warning_shown = False

    with ExitStack() as resources:
        gps_reader = None
        detection_store = None
        if args.gps_port:
            gps_reader = resources.enter_context(NmeaGpsReader(args.gps_port, args.gps_baud))
            detection_store = resources.enter_context(DetectionStore(args.database))
            print(f"Reading GPS from {args.gps_port}; saving detections to {args.database}.")

        inference = model.track if gps_reader else model.predict
        inference_options = {
            "source": source,
            "stream": True,
            "conf": args.conf,
            "imgsz": args.imgsz,
            "show": args.show,
            "save": True,
            "project": args.project,
            "name": args.name,
            "exist_ok": True,
        }
        if gps_reader:
            inference_options.update(persist=True, tracker="bytetrack.yaml")

        for frame_index, result in enumerate(inference(**inference_options)):
            boxes = result.boxes
            if boxes is None or not len(boxes):
                continue

            print(f"Frame {frame_index}: detected {len(boxes)} pothole object(s).")
            gps_fix = gps_reader.latest_fix(args.gps_max_age) if gps_reader else None
            track_ids = boxes.id
            for box_index, box in enumerate(boxes):
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                track_id = int(track_ids[box_index]) if track_ids is not None else None
                print(f"Object {box_index + 1}: conf={confidence:.3f}, bbox={bbox}")

                if not gps_reader:
                    continue
                if gps_fix is None:
                    if not gps_warning_shown:
                        print("Waiting for a recent GPS fix; detections without one are not saved.")
                        gps_warning_shown = True
                    continue
                if track_id is None:
                    if not tracking_warning_shown:
                        print("No tracker ID available; this detection will not be saved.")
                        tracking_warning_shown = True
                    continue
                if track_id in saved_track_ids:
                    continue

                if detection_store.save_track(run_id, track_id, source, gps_fix, confidence, bbox):
                    saved_track_ids.add(track_id)
                    print(
                        f"Saved track {track_id} at {gps_fix.latitude:.7f}, {gps_fix.longitude:.7f} "
                        "(camera GPS at first detection)."
                    )

    print(f"Inference completed for: {source}")


if __name__ == "__main__":
    main()
