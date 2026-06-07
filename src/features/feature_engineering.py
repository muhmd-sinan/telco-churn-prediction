import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# Fix TotalCharges — it has spaces, convert to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# --- Feature 1: engagement_score ---
# Count how many extra services the customer uses (0-6)
services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]
for col in services:
    df[col + "_flag"] = (df[col] == "Yes").astype(int)

df["engagement_score"] = df[[c + "_flag" for c in services]].sum(axis=1)

# Drop temp flag columns
df.drop(columns=[c + "_flag" for c in services], inplace=True)

# --- Feature 2: tenure_group ---
df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, float("inf")],
    labels=["new", "mid", "loyal", "champion"]
)

# --- Feature 3: monthly_per_tenure ---
# Monthly charge relative to how long they've been a customer
df["monthly_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

# --- Feature 4: is_month_to_month ---
df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

# --- Feature 5: has_support ---
df["has_support"] = (
    (df["TechSupport"] == "Yes") | (df["OnlineSecurity"] == "Yes")
).astype(int)

# --- Feature 6: responsiveness ---
# Digital-first customers: paperless billing + electronic payment
df["responsiveness"] = (
    (df["PaperlessBilling"] == "Yes").astype(int) +
    (df["PaymentMethod"] == "Electronic check").astype(int)
)

# Save
df.to_csv("data/processed/train_features.csv", index=False)
print(f"Saved to data/processed/train_features.csv")
print(f"New shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Quick correlation check
df["churn_binary"] = (df["Churn"] == "Yes").astype(int)
new_features = ["engagement_score", "monthly_per_tenure", "is_month_to_month",
                "has_support", "responsiveness"]
print("\nCorrelation with churn:")
for feat in new_features:
    corr = df[feat].corr(df["churn_binary"])
    print(f"  {feat}: {corr:.4f}")

df.drop(columns=["churn_binary"], inplace=True)
df.to_csv("data/processed/train_features.csv", index=False)
