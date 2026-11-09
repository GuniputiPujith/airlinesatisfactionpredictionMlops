import pandas as pd


def clean_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw passenger frame before prediction."""
    df = df.copy()
    df = df.drop(columns=["Unnamed: 0", "id"], errors="ignore")
    if "Arrival Delay in Minutes" in df.columns:
        df["Arrival Delay in Minutes"] = pd.to_numeric(df["Arrival Delay in Minutes"], errors="coerce")
    if "satisfaction" in df.columns:
        df = df.drop(columns=["satisfaction"])
    return df
