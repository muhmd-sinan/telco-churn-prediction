# Dataset Validation Findings - Telco Customer Churn

**Date**: 2026-06-07
**Dataset**: Telco-Customer-Churn.csv (IBM Watson Analytics)
**Status**: ✅ PASSED - Ready for ML Pipeline

---

## Dataset Shape
- **Rows**: 7,043
- **Columns**: 21
- **Result**: ✅ PASS

---

## Column Inventory

### Numerical Features
1. `tenure` - int64 (months as customer)
2. `MonthlyCharges` - float64
3. `TotalCharges` - object → converted to float64 (had spaces for new customers)
4. `SeniorCitizen` - int64 (0/1)

### Categorical Features
1. `customerID` - str (remove for modeling)
2. `gender` - str (Male/Female)
3. `Partner` - str (Yes/No)
4. `Dependents` - str (Yes/No)
5. `PhoneService` - str (Yes/No)
6. `MultipleLines` - str
7. `InternetService` - str (DSL/Fiber optic/No)
8. `OnlineSecurity` - str (Yes/No/No internet service)
9. `OnlineBackup` - str
10. `DeviceProtection` - str
11. `TechSupport` - str
12. `StreamingTV` - str
13. `StreamingMovies` - str
14. `Contract` - str (Month-to-month/One year/Two year)
15. `PaperlessBilling` - str (Yes/No)
16. `PaymentMethod` - str (4 methods)
17. `Churn` - str (TARGET: Yes/No)

---

## Data Quality

### Missing Values
- `TotalCharges`: 11 rows with blank strings (new customers with tenure=0)
- **Fix**: Convert to numeric, fill blanks with 0
- **Result**: ✅ Handled

### Target Distribution
- **Retained (No)**: 5,174 (73.5%)
- **Churned (Yes)**: 1,869 (26.5%)
- **Class imbalance**: Moderate (73:27). Handle with SMOTE or class_weight in Phase 5.

---

## Engineered Features (Phase 4)

| Feature | Formula | Correlation with Churn |
|---------|---------|----------------------|
| `engagement_score` | Count of Yes across 6 service columns | -0.09 |
| `tenure_group` | Cut tenure → new/mid/loyal/champion | — |
| `monthly_per_tenure` | MonthlyCharges / (tenure + 1) | +0.41 |
| `is_month_to_month` | Contract == 'Month-to-month' | +0.41 |
| `has_support` | TechSupport or OnlineSecurity == Yes | -0.18 |
| `responsiveness` | PaperlessBilling + Electronic check payment | +0.32 |

---

## Key Findings for Modeling

### Strongest Churn Predictors (from correlations)
1. `is_month_to_month` (+0.41) — no contract = easy to leave
2. `monthly_per_tenure` (+0.41) — high cost early on = churn
3. `responsiveness` (+0.32) — digital payment users churn more
4. `has_support` (-0.18) — support users stay longer
5. `InternetService` — Fiber optic users churn more than DSL

### Encoding Strategy
```
One-Hot: gender, Partner, Dependents, PhoneService, MultipleLines,
         InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
         TechSupport, StreamingTV, StreamingMovies, PaperlessBilling, PaymentMethod
Label/Ordinal: Contract (Month-to-month=0, One year=1, Two year=2)
               tenure_group (new=0, mid=1, loyal=2, champion=3)
Scale: tenure, MonthlyCharges, TotalCharges, monthly_per_tenure, engagement_score
Remove: customerID
```

---

## Next Steps
1. ✅ Phase 1 COMPLETE
2. ✅ Phase 2 SKIPPED (minimal cleaning — only TotalCharges fix)
3. ⏳ Phase 3: EDA (pending Power BI)
4. ✅ Phase 4: Feature Engineering COMPLETE
5. → Phase 5: Preprocessing — encoding, scaling, train/test split, SMOTE
