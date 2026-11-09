import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.preprocess import load_frame


def run_evaluation():
    print("Starting Model Evaluation...")
    os.makedirs("outputs", exist_ok=True)

    X_test_final = np.load("data/processed/X_test_final.npy")
    y_test = np.load("data/processed/y_test.npy")
    model = joblib.load("models/random_forest_baseline.pkl")
    y_pred = model.predict(X_test_final)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n--- Model Evaluation Report ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("-------------------------------\n")

    df = load_frame()
    X = df.drop("satisfaction", axis=1)
    y = df["satisfaction"]
    _, X_test_raw, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    errors_df = X_test_raw.copy()
    errors_df["Actual_Satisfaction"] = y_test
    errors_df["Predicted_Satisfaction"] = y_pred
    false_negatives = errors_df[(errors_df["Actual_Satisfaction"] == 1) & (errors_df["Predicted_Satisfaction"] == 0)]
    false_positives = errors_df[(errors_df["Actual_Satisfaction"] == 0) & (errors_df["Predicted_Satisfaction"] == 1)]
    false_negatives.to_csv("outputs/false_negatives.csv", index=False)
    false_positives.to_csv("outputs/false_positives.csv", index=False)
    print("Evaluation complete and error analysis files saved!")


if __name__ == "__main__":
    run_evaluation()
