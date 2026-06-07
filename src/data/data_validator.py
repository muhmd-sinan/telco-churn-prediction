"""
Dataset Validation Script - Netflix Churn Prediction
Validates dataset quality and generates comprehensive report
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path


def validate_dataset(filepath):
    """Load and validate dataset with comprehensive checks"""
    
    print("=" * 70)
    print("NETFLIX CHURN PREDICTION - DATASET VALIDATION REPORT")
    print("=" * 70)
    
    # Load dataset
    df = pd.read_csv(filepath)
    
    # ========== SHAPE VALIDATION ==========
    print("\n" + "=" * 70)
    print("1. DATASET SHAPE")
    print("=" * 70)
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    
    expected_rows = 50000
    expected_cols = 20
    
    if df.shape[0] == expected_rows and df.shape[1] == expected_cols:
        print(f"✅ PASS: Shape matches expected ({expected_rows:,} rows × {expected_cols} columns)")
    else:
        print(f"⚠️  WARNING: Expected {expected_rows:,} rows × {expected_cols} columns")
    
    # ========== COLUMN VALIDATION ==========
    print("\n" + "=" * 70)
    print("2. COLUMN VALIDATION")
    print("=" * 70)
    print(f"Total columns: {len(df.columns)}")
    print("\nColumns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    expected_columns = [
        'user_id', 'age', 'gender', 'country', 'account_age_months',
        'subscription_type', 'monthly_fee', 'payment_method', 'primary_device',
        'devices_used', 'favorite_genre', 'avg_watch_time_minutes',
        'watch_sessions_per_week', 'binge_watch_sessions', 'completion_rate',
        'rating_given', 'content_interactions', 'recommendation_click_rate',
        'days_since_last_login', 'churned'
    ]
    
    missing_cols = set(expected_columns) - set(df.columns)
    extra_cols = set(df.columns) - set(expected_columns)
    
    if not missing_cols and not extra_cols:
        print("\n✅ PASS: All expected columns present")
    else:
        if missing_cols:
            print(f"\n❌ MISSING COLUMNS: {missing_cols}")
        if extra_cols:
            print(f"\n⚠️  EXTRA COLUMNS: {extra_cols}")
    
    # ========== DATA TYPES ==========
    print("\n" + "=" * 70)
    print("3. DATA TYPES")
    print("=" * 70)
    print(df.dtypes.to_string())
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nNumerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    # ========== MISSING VALUES ==========
    print("\n" + "=" * 70)
    print("4. MISSING VALUES")
    print("=" * 70)
    
    missing = df.isnull().sum()
    total_missing = missing.sum()
    
    if total_missing == 0:
        print("✅ PASS: No missing values detected")
    else:
        print(f"⚠️  TOTAL MISSING: {total_missing:,}")
        missing_pct = (missing / len(df) * 100).round(2)
        missing_report = pd.DataFrame({
            'Missing Count': missing[missing > 0],
            'Missing %': missing_pct[missing_pct > 0]
        })
        print(missing_report)
    
    # ========== DUPLICATES ==========
    print("\n" + "=" * 70)
    print("5. DUPLICATE RECORDS")
    print("=" * 70)
    
    exact_duplicates = df.duplicated().sum()
    user_id_duplicates = df['user_id'].duplicated().sum()
    
    print(f"Exact duplicates: {exact_duplicates}")
    print(f"Duplicate user_ids: {user_id_duplicates}")
    
    if exact_duplicates == 0 and user_id_duplicates == 0:
        print("✅ PASS: No duplicate records")
    else:
        print("⚠️  WARNING: Duplicates detected - requires cleaning")
    
    # ========== TARGET DISTRIBUTION ==========
    print("\n" + "=" * 70)
    print("6. TARGET VARIABLE DISTRIBUTION")
    print("=" * 70)
    
    target_counts = df['churned'].value_counts()
    churn_rate = (df['churned'] == 'Yes').sum() / len(df) * 100
    
    print(f"Retained (No): {target_counts.get('No', 0):,} ({target_counts.get('No', 0)/len(df)*100:.2f}%)")
    print(f"Churned (Yes): {target_counts.get('Yes', 0):,} ({target_counts.get('Yes', 0)/len(df)*100:.2f}%)")
    print(f"\nChurn Rate: {churn_rate:.2f}%")
    
    if 15 <= churn_rate <= 25:
        print(f"✅ PASS: Churn rate within expected range (15-25%)")
    else:
        print(f"⚠️  WARNING: Churn rate outside expected range (15-25%)")
    
    # ========== STATISTICAL SUMMARY ==========
    print("\n" + "=" * 70)
    print("7. STATISTICAL SUMMARY (Numerical Features)")
    print("=" * 70)
    print(df.describe().round(2).to_string())
    
    # ========== BUSINESS RULE VALIDATION ==========
    print("\n" + "=" * 70)
    print("8. BUSINESS RULE VALIDATION")
    print("=" * 70)
    
    validation_rules = {
        'age': (18, 100, "Minimum 18 for streaming service"),
        'account_age_months': (0, 120, "Max 10 years"),
        'monthly_fee': [7.99, 12.99, 15.99],  # Valid subscription tiers
        'completion_rate': (0, 100, "Percentage range"),
        'rating_given': (1, 5, "Rating scale 1-5"),
        'days_since_last_login': (0, 365, "Max 1 year")
    }
    
    all_valid = True
    
    for col, rule in validation_rules.items():
        if col not in df.columns:
            continue
            
        if isinstance(rule, tuple):
            min_val, max_val, reason = rule
            invalid = df[(df[col] < min_val) | (df[col] > max_val)]
            if len(invalid) == 0:
                print(f"✅ {col}: Valid range [{min_val}, {max_val}] - {reason}")
            else:
                print(f"❌ {col}: {len(invalid)} invalid values (expected [{min_val}, {max_val}]) - {reason}")
                all_valid = False
        
        elif isinstance(rule, list):
            valid_tiers = rule
            invalid = df[~df[col].isin(valid_tiers)]
            if len(invalid) == 0:
                print(f"✅ {col}: Valid values {valid_tiers}")
            else:
                print(f"❌ {col}: {len(invalid)} invalid values (expected {valid_tiers})")
                all_valid = False
    
    # ========== OUTLIER DETECTION ==========
    print("\n" + "=" * 70)
    print("9. OUTLIER DETECTION (IQR Method)")
    print("=" * 70)
    
    continuous_features = [
        'avg_watch_time_minutes', 'watch_sessions_per_week',
        'binge_watch_sessions', 'content_interactions',
        'recommendation_click_rate', 'completion_rate'
    ]
    
    for col in continuous_features:
        if col not in df.columns:
            continue
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_pct = len(outliers) / len(df) * 100
        
        status = "⚠️ " if outlier_pct > 5 else "✅"
        print(f"{status} {col}: {len(outliers):,} outliers ({outlier_pct:.2f}%)")
    
    # ========== CATEGORICAL VALUE COUNTS ==========
    print("\n" + "=" * 70)
    print("10. CATEGORICAL VALUE DISTRIBUTIONS")
    print("=" * 70)
    
    categorical_features = ['gender', 'subscription_type', 'payment_method', 'favorite_genre', 'primary_device']
    
    for col in categorical_features:
        if col not in df.columns:
            continue
        
        print(f"\n{col}:")
        value_counts = df[col].value_counts()
        for val, count in value_counts.items():
            pct = count / len(df) * 100
            print(f"  - {val}: {count:,} ({pct:.2f}%)")
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    checks_passed = []
    checks_passed.append(df.shape[0] == expected_rows and df.shape[1] == expected_cols)
    checks_passed.append(len(missing_cols) == 0)
    checks_passed.append(total_missing == 0)
    checks_passed.append(exact_duplicates == 0)
    checks_passed.append(user_id_duplicates == 0)
    checks_passed.append(15 <= churn_rate <= 25)
    checks_passed.append(all_valid)
    
    total_checks = len(checks_passed)
    passed_checks = sum(checks_passed)
    
    print(f"\nChecks Passed: {passed_checks}/{total_checks}")
    
    if all(checks_passed):
        print("\n✅ DATASET VALIDATION: PASSED")
        print("✅ Data quality meets requirements for ML pipeline")
        print("✅ Ready to proceed to Phase 2: Data Cleaning")
    else:
        print("\n⚠️  DATASET VALIDATION: ISSUES DETECTED")
        print("⚠️  Review warnings above before proceeding")
    
    print("=" * 70)
    
    return df


if __name__ == "__main__":
    filepath = "data/raw/netflix_user_behavior_dataset.csv"
    df = validate_dataset(filepath)
