import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_frame(path="data/raw/airline.csv"):
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0", "id"], errors="ignore")
    df["Arrival Delay in Minutes"] = pd.to_numeric(df["Arrival Delay in Minutes"], errors="coerce")
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Arrival Delay in Minutes"].median())
    df["satisfaction"] = df["satisfaction"].apply(
        lambda x: 1 if str(x).strip().lower() == "satisfied" else 0
    ).astype(int)
    return df


def run_preprocessing():
    print("Starting Preprocessing Pipeline...")
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = load_frame()
    X = df.drop("satisfaction", axis=1)
    y = df["satisfaction"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    num_cols = X_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train[num_cols])
    x_test_scaled = scaler.transform(X_test[num_cols])

    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    x_train_encoded = ohe.fit_transform(X_train[cat_cols])
    x_test_encoded = ohe.transform(X_test[cat_cols])

    X_train_final = np.hstack((x_train_scaled, x_train_encoded))
    X_test_final = np.hstack((x_test_scaled, x_test_encoded))

    np.save("data/processed/X_train_final.npy", X_train_final)
    np.save("data/processed/X_test_final.npy", X_test_final)
    np.save("data/processed/y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save("data/processed/y_test.npy", y_test.to_numpy(dtype=np.int64))
    X_train.to_csv("data/processed/X_train_raw.csv", index=False)
    X_test.to_csv("data/processed/X_test_raw.csv", index=False)

    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(ohe, "models/ohe.pkl")

    metadata = {
        "dataset_name": "Airline Passenger Satisfaction",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": list(num_cols),
        "categorical_features": list(cat_cols),
        "target": "satisfaction",
        "positive_class": "satisfied",
    }
    with open("data/processed/dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    run_preprocessing()
