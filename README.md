# GeoPothole

This project is a starter pothole detection pipeline for a 4th-year road monitoring system. It is designed to be extended for mobile reporting, dashcam detection, geotagging, and city-level pothole density scoring.

## Project goal

- Detect potholes from images or video frames
- Save geotagged detection records
- Rank cities by pothole count per square kilometer
- Extend this into a full crowdsourced road-monitoring platform

## Tech stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- pandas
- FastAPI (planned for backend)
- PostgreSQL + PostGIS (planned for geospatial database)

## Structure

- `src/train_model.py` - train a pothole detection YOLO model
- `src/detect_potholes.py` - run inference on an image/video/folder
- `src/city_ranking.py` - compute city pothole density per sq km
- `data/` - example datasets and city area references

## Quick start

1. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

2. Train a model:
   ```bash
   python src/train_model.py --data data/pothole_dataset.yaml --epochs 20 --imgsz 640
   ```

3. Run inference:
   ```bash
   python src/detect_potholes.py --source path/to/image.jpg --weights runs/train/exp/weights/best.pt
   ```

4. Rank cities by pothole density:
   ```bash
   python src/city_ranking.py --input data/example_city_reports.csv --area-file data/city_area_reference.csv
   ```

## Notes

- The initial model is a practical starter for the minor project.
- Later, you can add duplicate pothole clustering, severity scoring, and a dashboard.

---

This repository is meant to be the foundation of your minor project and can evolve into the full major-project system.
