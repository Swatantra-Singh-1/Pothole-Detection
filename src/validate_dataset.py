import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a YOLO-style pothole dataset structure.")
    parser.add_argument("--dataset-dir", type=str, default="data", help="Root folder containing the dataset.")
    return parser.parse_args()


def validate_dataset(dataset_root: Path):
    required = {
        "images/train": dataset_root / "images" / "train",
        "images/val": dataset_root / "images" / "val",
        "labels/train": dataset_root / "labels" / "train",
        "labels/val": dataset_root / "labels" / "val",
        "yaml": dataset_root / "pothole_dataset.yaml",
    }

    print("Checking dataset structure...\n")
    for name, path in required.items():
        if path.exists():
            print(f"[OK] {name}: {path}")
        else:
            print(f"[MISSING] {name}: {path}")

    train_images = list((required["images/train"]).glob("*"))
    val_images = list((required["images/val"]).glob("*"))
    train_labels = list((required["labels/train"]).glob("*"))
    val_labels = list((required["labels/val"]).glob("*"))

    print("\nImage and label summary:")
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")
    print(f"Train labels: {len(train_labels)}")
    print(f"Val labels: {len(val_labels)}")

    if not required["yaml"].exists():
        print("\nDataset YAML file is missing. Create it before training.")
        return

    if not train_images:
        print("\nNo training images found yet. Add images to data/images/train.")
    if not val_images:
        print("\nNo validation images found yet. Add images to data/images/val.")
    if not train_labels and train_images:
        print("\nTraining images exist but labels are missing. Annotate them in YOLO format.")
    if not val_labels and val_images:
        print("\nValidation images exist but labels are missing. Annotate them in YOLO format.")

    if train_images and val_images and train_labels and val_labels:
        print("\nDataset structure looks ready for training.")


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_dir)
    validate_dataset(dataset_root)


if __name__ == "__main__":
    main()
