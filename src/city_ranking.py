import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Rank cities by pothole count per square kilometer.")
    parser.add_argument("--input", type=str, required=True, help="CSV file with city reports.")
    parser.add_argument(
        "--area-file",
        type=str,
        default=None,
        help="Optional CSV file with city and area_sq_km columns.",
    )
    return parser.parse_args()


def compute_density(df: pd.DataFrame, area_lookup: dict | None = None) -> pd.DataFrame:
    if not {"city"}.issubset(df.columns):
        raise ValueError("CSV must contain a 'city' column.")

    if "pothole_count" in df.columns:
        city_df = df.groupby("city", as_index=False)["pothole_count"].sum()
        if "area_sq_km" in df.columns:
            city_df["area_sq_km"] = df.groupby("city")["area_sq_km"].first().values
        else:
            city_df["area_sq_km"] = city_df["city"].map(area_lookup or {})
    else:
        city_df = df.groupby("city").size().reset_index(name="pothole_count")
        city_df["area_sq_km"] = city_df["city"].map(area_lookup or {})

    city_df = city_df.dropna(subset=["pothole_count", "area_sq_km"]).copy()
    city_df["potholes_per_sq_km"] = city_df["pothole_count"] / city_df["area_sq_km"]
    city_df = city_df.sort_values("potholes_per_sq_km", ascending=False).reset_index(drop=True)
    return city_df


def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    area_lookup = {}

    if args.area_file:
        area_path = Path(args.area_file)
        if area_path.exists():
            area_df = pd.read_csv(area_path)
            if {"city", "area_sq_km"}.issubset(area_df.columns):
                area_lookup = dict(zip(area_df["city"], area_df["area_sq_km"]))

    ranked = compute_density(df, area_lookup)
    print(ranked.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
