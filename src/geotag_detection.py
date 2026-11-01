import argparse
import csv
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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


@dataclass(frozen=True)
class GpsFix:
    latitude: float
    longitude: float
    observed_at: datetime


class NmeaGpsReader:
    def __init__(self, port: str, baud_rate: int = 9600):
        try:
            import pynmea2
            import serial
        except ImportError as exc:
            raise RuntimeError("GPS input requires pyserial and pynmea2; install requirements.txt.") from exc

        self._pynmea2 = pynmea2
        self._serial = serial.Serial(port, baudrate=baud_rate, timeout=1)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._latest_fix: GpsFix | None = None
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while not self._stop_event.is_set():
            line = self._serial.readline().decode("ascii", errors="ignore").strip()
            if not line:
                continue
            try:
                message = self._pynmea2.parse(line)
                if message.sentence_type != "GGA" or int(message.gps_qual or 0) <= 0:
                    continue
                fix = GpsFix(
                    latitude=float(message.latitude),
                    longitude=float(message.longitude),
                    observed_at=datetime.now(timezone.utc),
                )
            except (self._pynmea2.ParseError, TypeError, ValueError):
                continue
            with self._lock:
                self._latest_fix = fix

    def latest_fix(self, max_age_seconds: float = 5.0) -> GpsFix | None:
        with self._lock:
            fix = self._latest_fix
        if fix is None or datetime.now(timezone.utc) - fix.observed_at > timedelta(seconds=max_age_seconds):
            return None
        return fix

    def close(self):
        self._stop_event.set()
        self._thread.join(timeout=2)
        self._serial.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class DetectionStore:
    def __init__(self, database_path: str):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.database_path)
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS pothole_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                detected_at TEXT NOT NULL,
                gps_fix_at TEXT NOT NULL,
                source TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                confidence REAL NOT NULL,
                x_min REAL NOT NULL,
                y_min REAL NOT NULL,
                x_max REAL NOT NULL,
                y_max REAL NOT NULL,
                map_url TEXT NOT NULL,
                location_method TEXT NOT NULL,
                UNIQUE (run_id, track_id)
            )"""
        )
        self._connection.commit()

    def save_track(self, run_id: str, track_id: int, source: str, fix: GpsFix, confidence: float, bbox: list[float]) -> bool:
        latitude = fix.latitude
        longitude = fix.longitude
        map_url = f"https://www.google.com/maps/search/?api=1&query={latitude:.7f},{longitude:.7f}"
        cursor = self._connection.execute(
            """INSERT OR IGNORE INTO pothole_detections (
                run_id, track_id, detected_at, gps_fix_at, source, latitude, longitude,
                confidence, x_min, y_min, x_max, y_max, map_url, location_method
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                track_id,
                datetime.now(timezone.utc).isoformat(),
                fix.observed_at.isoformat(),
                source,
                latitude,
                longitude,
                confidence,
                *bbox,
                map_url,
                "camera_gps_at_first_detection",
            ),
        )
        self._connection.commit()
        return cursor.rowcount == 1

    def close(self):
        self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


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
