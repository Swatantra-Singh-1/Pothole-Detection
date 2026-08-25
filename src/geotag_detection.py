import argparse
import csv
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Save a geotagged pothole detection record as CSV.")
    parser.add_argument("--image", type=str, required=True, help="Path to the captured image.")
    parser.add_argument("--latitude", type=float, required=True, help="Latitude of the pothole.")
    parser.add_argument("--longitude", type=float, required=True, help="Longitude of the pothole.")
    parser.add_argument("--confidence", type=float, default=0.0, help="Detection confidence score.")
    parser.add_argument("--city", type=str, default="unknown", help="City or location name.")
    parser.add_argument("--output", type=str, default="data/geotagged_detections.csv", help="CSV file to store records.")
    return parser.parse_args()


def append_detection_record(image_path: str, latitude: float, longitude: float, confidence: float, city: str, output_path: str):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["timestamp", "city", "image_path", "latitude", "longitude", "confidence"]
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "city": city,
        "image_path": image_path,
        "latitude": latitude,
        "longitude": longitude,
        "confidence": confidence,
    }

    file_exists = output_file.exists()
    with output_file.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Saved geotagged detection record to: {output_file}")
    print(row)


def main():
    args = parse_args()
    append_detection_record(
        image_path=args.image,
        latitude=args.latitude,
        longitude=args.longitude,
        confidence=args.confidence,
        city=args.city,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
