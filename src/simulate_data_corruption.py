import pandas as pd
from src.validate_data import validate_schema


def simulate_corrupted_data(df):
    """Simulates real-world data corruption scenarios across multiple columns."""
    print("[INFO] Injecting simulated faults into data...")
    corrupted_df = df.copy()
    corrupted_df.loc[0, "Age"] = -5
    corrupted_df.loc[1, "Class"] = "First"
    corrupted_df.loc[2, "satisfaction"] = "Maybe"
    corrupted_df.loc[3, "Flight Distance"] = "Very Far"
    corrupted_df["unexpected_column"] = 123
    corrupted_df = corrupted_df.drop(columns=["Gender"])
    return corrupted_df


if __name__ == "__main__":
    data_path = "data/raw/airline.csv"
    try:
        raw_df = pd.read_csv(data_path)
        corrupted_df = simulate_corrupted_data(raw_df)
        print("\n--- Running Validation on Corrupted Data ---")
        validate_schema(corrupted_df, output_report_name="corrupted_simulation_errors.csv")
    except FileNotFoundError:
        print(f"[ERROR] Could not find data file at {data_path}.")
