from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_DATASET = DATA_DIR / "pothole_dataset.yaml"

# Example city area reference (km²) used for density ranking
CITY_AREAS = {
    "Delhi": 1484,
    "Ghaziabad": 210,
    "Lucknow": 631,
    "Noida": 203,
    "Bengaluru": 741,
    "Hyderabad": 650,
    "Pune": 496,
}
