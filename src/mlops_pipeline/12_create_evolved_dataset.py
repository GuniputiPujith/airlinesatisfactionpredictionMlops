
import pandas as pd
import numpy as np
from pathlib import Path

SOURCE_PATH = Path("data/processed/X_test_raw.csv")
OUTPUT_PATH = Path("data/monitoring/drift/evolved_data.csv")

RANDOM_SEED = 42


def create_evolved_dataset():

    rng = np.random.default_rng(RANDOM_SEED)

    df = pd.read_csv(SOURCE_PATH).copy()

    # Numerical drift: Flight Distance
    df["Flight Distance"] = (
        df["Flight Distance"] *
        rng.normal(1.15, 0.08, len(df))
    ).clip(lower=0)

    # Numerical drift: Age
    df["Age"] = (
        df["Age"] + rng.normal(4, 6, len(df))
    ).clip(lower=0, upper=100).round().astype(int)

    # Numerical drift: Arrival Delay
    df["Arrival Delay in Minutes"] = (
        df["Arrival Delay in Minutes"] *
        rng.normal(1.25, 0.1, len(df))
    ).clip(lower=0).round(2)

    # Categorical drift: Class
    df["Class"] = rng.choice(
        ["Business", "Eco", "Eco Plus"],
        size=len(df),
        p=[0.20, 0.70, 0.10]
    )

    # Categorical drift: Type of Travel
    df["Type of Travel"] = rng.choice(
        ["Business travel", "Personal Travel"],
        size=len(df),
        p=[0.35, 0.65]
    )

    # Categorical drift: Customer Type
    df["Customer Type"] = rng.choice(
        ["Loyal Customer", "disloyal Customer"],
        size=len(df),
        p=[0.45, 0.55]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nEVOLVED AIRLINE DATASET CREATED")
    print("--------------------------------")
    print("Source:", SOURCE_PATH)
    print("Output:", OUTPUT_PATH)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nModified Airline Features:")
    print("- Flight Distance")
    print("- Age")
    print("- Arrival Delay in Minutes")
    print("- Class")
    print("- Type of Travel")
    print("- Customer Type")


if __name__ == "__main__":
    create_evolved_dataset()
