import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create a YOLO-style pothole dataset structure.")
    parser.add_argument("--dataset-dir", type=str, default="data", help="Dataset root folder to create.")
    return parser.parse_args()


def create_dataset_structure(root: Path):
    folders = [
        root / "images" / "train",
        root / "images" / "val",
        root / "labels" / "train",
        root / "labels" / "val",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    yaml_content = (
        "path: " + str(root) + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['pothole']\n"
    )

    (root / "pothole_dataset.yaml").write_text(yaml_content, encoding="utf-8")
    (root / "README.md").write_text(
        "# Pothole dataset\n\n"
        "Place training and validation images into images/train and images/val.\n"
        "Place matching YOLO label files in labels/train and labels/val.\n",
        encoding="utf-8",
    )

    print(f"Dataset structure created at: {root}")
    print("Folders created:")
    for folder in folders:
        print(" -", folder)


def main():
    args = parse_args()
    root = Path(args.dataset_dir)
    create_dataset_structure(root)


if __name__ == "__main__":
    main()
