"""
Future shelf-life training entry point.

This script intentionally refuses to train until real longitudinal labels exist.
That prevents accidentally training a model on made-up "days remaining" values.
"""

from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {
    "image_path",
    "temperature_c",
    "humidity_pct",
    "days_left",
}


def main():
    csv_path = Path("data/longitudinal/shelf_life.csv")

    if not csv_path.exists():
        raise SystemExit(
            "No shelf-life dataset found.\n"
            "Create data/longitudinal/shelf_life.csv with real observations.\n"
            "See shelf_life/README.md for the schema."
        )

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["days_left"].isna().any():
        raise SystemExit("days_left contains missing labels.")

    print(f"Loaded {len(df)} real shelf-life observations.")
    print(df.head())

    # TODO:
    # 1. Generate CNN embeddings for each image.
    # 2. Combine embeddings with temperature/humidity.
    # 3. Split by banana identity, NOT randomly by image.
    # 4. Train a regression model.
    # 5. Evaluate MAE/RMSE.
    # 6. Save the model.
    #
    # We leave this incomplete on purpose until real longitudinal
    # data exists. A fake target would make the experiment meaningless.


if __name__ == "__main__":
    main()
