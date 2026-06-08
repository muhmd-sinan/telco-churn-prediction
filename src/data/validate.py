import pandas as pd

DATA_PATH = "data/raw/Telco-Customer-Churn.csv"

EXPECTED_ROWS = 7043
EXPECTED_COLS = 21
EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
]


def validate():
    df = pd.read_csv(DATA_PATH)

    # Check row count
    assert len(df) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {len(df)}"

    # Check columns
    assert list(df.columns) == EXPECTED_COLUMNS, f"Column mismatch: {list(df.columns)}"

    # Check TotalCharges is string with blanks (known issue in raw data)
    blank_charges = df[df["TotalCharges"].str.strip() == ""].shape[0]
    print(f"Blank TotalCharges: {blank_charges} (expected 11)")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Validation passed.")


if __name__ == "__main__":
    validate()
