# Telco Customer Churn Prediction

A machine learning system that predicts customer churn for a telecom provider, with an interactive analytics dashboard built in Streamlit.

## What It Does

- Trains three ML models (Logistic Regression, Random Forest, XGBoost) on 7,043 customer records
- Evaluates each model on a held-out test set with full metrics: AUC-ROC, precision, recall, F1, confusion matrix
- Provides a four-page interactive dashboard for exploration, model comparison, and per-model deep-dive

## Dashboard Pages

| Page | What you get |
|------|-------------|
| **Overview** | KPI cards, churn donut chart, contract/tenure/payment method breakdowns |
| **CSV Explorer** | Upload any CSV — or use the built-in dataset — for distributions, correlation heatmap, and scatter plots |
| **Model Performance** | Side-by-side ROC curves, Precision-Recall curves, prediction score distributions, full comparison table |
| **Model Deep-Dive** | Per-model confusion matrix, classification report with inline progress bars, threshold analysis, feature importance/coefficients, risk bucket breakdown |

## Model Results

All models were evaluated on a 20% held-out test set (1,409 customers).

### Best Model — Logistic Regression

| Metric | Score |
|--------|-------|
| AUC-ROC | **0.8415** |
| Avg Precision | 0.6299 |
| Accuracy | 80.91% |
| Precision (Churn) | 67.1% |
| Recall (Churn) | 55.1% |
| F1-Score (Churn) | 60.5% |

Confusion matrix: TN=934 · FP=101 · FN=168 · TP=206

Logistic Regression outperforms the tree-based models on AUC-ROC, accuracy, and all churn-class metrics despite being the simplest model — a common result on tabular datasets with strong linear separability.

### Full Comparison

| Model | AUC-ROC | Accuracy | Precision | Recall | F1 | Avg Precision |
|-------|---------|----------|-----------|--------|----|---------------|
| **Logistic Regression** ⭐ | **0.8415** | **80.91%** | **67.1%** | **55.1%** | **60.5%** | **62.99%** |
| Random Forest | 0.8216 | 77.50% | 59.5% | 47.9% | 53.0% | 61.45% |
| XGBoost | 0.8203 | 77.50% | 58.9% | 50.5% | 54.4% | 60.91% |

> **Metrics are for the Churn (positive) class** — the harder and more business-relevant class to predict correctly.

### Confusion Matrices

| Model | TN | FP | FN | TP |
|-------|----|----|----|----|
| Logistic Regression | 934 | 101 | 168 | 206 |
| Random Forest | 913 | 122 | 195 | 179 |
| XGBoost | 903 | 132 | 185 | 189 |

## Project Structure

```
.
├── app.py                    # Streamlit dashboard (entry point)
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── models/
│   ├── scaler.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── xgboost.joblib
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
├── api/
├── tests/
├── requirements.txt
└── ROADMAP.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

The app expects to be run from the project root so that the relative paths to `data/` and `models/` resolve correctly.

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data | pandas, numpy |
| ML | scikit-learn, xgboost |
| Dashboard | Streamlit, Plotly |
| Serialisation | joblib |

## Dataset

**Telco-Customer-Churn.csv** — 7,043 customers, 21 features including tenure, contract type, internet service, payment method, monthly/total charges, and the binary churn label (`Yes`/`No`).

Class imbalance: ~26.5% churn.

## Key Findings

- Month-to-month customers churn at ~43 % vs 3 % for two-year contracts
- Fiber optic internet service has the highest churn rate among all service types
- Customers in their first 12 months are 3× more likely to churn than long-tenure customers
- Electronic check payers churn significantly more than customers using other payment methods
