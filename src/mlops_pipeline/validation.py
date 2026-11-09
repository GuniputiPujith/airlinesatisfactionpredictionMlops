import os
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check

RATING = Check.in_range(0, 5)


def get_airline_schema():
    """Strict schema for the Airline Passenger Satisfaction dataset."""
    return pa.DataFrameSchema({
        "Unnamed: 0": Column(pa.Int),
        "id": Column(pa.Int),
        "Gender": Column(pa.String, Check.isin(["Male", "Female"])),
        "Customer Type": Column(pa.String, Check.isin(["Loyal Customer", "disloyal Customer"])),
        "Age": Column(pa.Int, Check.in_range(0, 120)),
        "Type of Travel": Column(pa.String, Check.isin(["Business travel", "Personal Travel"])),
        "Class": Column(pa.String, Check.isin(["Business", "Eco", "Eco Plus"])),
        "Flight Distance": Column(pa.Int, Check.ge(0)),
        "Inflight wifi service": Column(pa.Int, RATING),
        "Departure/Arrival time convenient": Column(pa.Int, RATING),
        "Ease of Online booking": Column(pa.Int, RATING),
        "Gate location": Column(pa.Int, RATING),
        "Food and drink": Column(pa.Int, RATING),
        "Online boarding": Column(pa.Int, RATING),
        "Seat comfort": Column(pa.Int, RATING),
        "Inflight entertainment": Column(pa.Int, RATING),
        "On-board service": Column(pa.Int, RATING),
        "Leg room service": Column(pa.Int, RATING),
        "Baggage handling": Column(pa.Int, RATING),
        "Checkin service": Column(pa.Int, RATING),
        "Inflight service": Column(pa.Int, RATING),
        "Cleanliness": Column(pa.Int, RATING),
        "Departure Delay in Minutes": Column(pa.Int, Check.ge(0)),
        "Arrival Delay in Minutes": Column(pa.Float, Check.ge(0.0), nullable=True),
        "satisfaction": Column(pa.String, Check.isin(["satisfied", "neutral or dissatisfied"])),
    }, strict=True)


def get_telco_schema():
    """Backward-compatible name used by older lab imports."""
    return get_airline_schema()


def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    schema = get_airline_schema()
    try:
        schema.validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True
    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED. Corruptions detected.")
        failures = err.failure_cases[["schema_context", "column", "check", "failure_case", "index"]]
        print(failures.head(30).to_string())
        os.makedirs("artifacts", exist_ok=True)
        report_path = os.path.join("artifacts", output_report_name)
        failures.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_path}'.")
        return False


if __name__ == "__skip_main__":
    data_path = "data/raw/airline.csv"
    if os.path.exists(data_path):
        raw_df = pd.read_csv(data_path)
        ok = validate_schema(raw_df, output_report_name="baseline_validation.csv")
        raise SystemExit(0 if ok else 1)
    print(f"[ERROR] Target file not found at: {data_path}")
    raise SystemExit(1)
