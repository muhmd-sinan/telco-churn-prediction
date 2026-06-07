# Telecom Customer Churn Prediction

End-to-end ML pipeline to predict customer churn using the IBM Telco Customer Churn dataset.

## Dataset
- **Source**: IBM Watson Analytics — Telco Customer Churn (Kaggle)
- **File**: `data/raw/Telco-Customer-Churn.csv`
- **Size**: 7,043 rows × 21 columns
- **Target**: `Churn` (Yes/No) — ~26.5% churn rate

## Goal
Predict which customers are likely to churn so the business can intervene early with retention offers.

**Targets**: Recall >80%, ROC-AUC >85%

## Project Structure
```
├── data/
│   ├── raw/                  # Original dataset
│   ├── processed/            # Engineered + preprocessed data
│   └── predictions/          # Model output
├── src/
│   ├── data/                 # Validation scripts
│   ├── features/             # Feature engineering
│   └── models/               # Training + evaluation
├── reports/
│   └── figures/              # Visualizations
├── models/                   # Saved model files
└── requirements.txt
```

## Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Progress
- ✅ Phase 1: Environment + project structure
- ✅ Phase 4: Feature engineering (6 new features)
- ⏳ Phase 5: Preprocessing
- ⏳ Phase 6: Model training
- ⏳ Phase 7-10: Evaluation, SHAP, segmentation, deployment

## Engineered Features
| Feature | Description |
|---------|-------------|
| `engagement_score` | Count of extra services subscribed (0-6) |
| `tenure_group` | Tenure bucketed: new / mid / loyal / champion |
| `monthly_per_tenure` | Monthly charge relative to tenure |
| `is_month_to_month` | 1 if no long-term contract |
| `has_support` | 1 if TechSupport or OnlineSecurity active |
| `responsiveness` | Paperless billing + electronic payment score |

## Key Correlations with Churn
- `is_month_to_month`: +0.41
- `monthly_per_tenure`: +0.41
- `responsiveness`: +0.32
- `has_support`: -0.18
- `engagement_score`: -0.09
