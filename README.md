# Telco Customer Churn Prediction

Machine learning system that predicts customer churn for a telecom provider, with an interactive analytics dashboard built in Streamlit.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run from the project root so relative paths to `data/` and `models/` resolve correctly.

## Dashboard

Four pages, dark navy theme, fully interactive Plotly charts:

| Page | Contents |
|------|----------|
| **Overview** | KPI cards, churn donut, contract / tenure / payment breakdowns, insight summary |
| **CSV Explorer** | Upload any CSV or use the built-in dataset — distributions, correlation heatmap, scatter explorer |
| **Model Performance** | ROC curves, Precision-Recall curves, score distributions, full metric comparison table |
| **Model Deep-Dive** | Confusion matrix, classification report, threshold analysis, feature importance, risk buckets |

## Model Results

Evaluated on a 20% held-out test set (1,409 customers).

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

### Full Comparison

| Model | AUC-ROC | Accuracy | Precision | Recall | F1 | Avg Precision |
|-------|---------|----------|-----------|--------|----|---------------|
| **Logistic Regression** ⭐ | **0.8415** | **80.91%** | **67.1%** | **55.1%** | **60.5%** | 62.99% |
| Random Forest | 0.8216 | 77.50% | 59.5% | 47.9% | 53.0% | 61.45% |
| XGBoost | 0.8203 | 77.50% | 58.9% | 50.5% | 54.4% | 60.91% |

> Metrics are for the Churn (positive) class. Logistic Regression wins on all metrics — common on tabular data with strong linear separability.

### Confusion Matrices

| Model | TN | FP | FN | TP |
|-------|----|----|----|----|
| Logistic Regression | 934 | 101 | 168 | 206 |
| Random Forest | 913 | 122 | 195 | 179 |
| XGBoost | 903 | 132 | 185 | 189 |

## Key Findings

- Month-to-month customers churn at ~43% vs 3% for two-year contracts
- Fiber optic internet has the highest churn rate across all service types
- Customers in their first 12 months are 3× more likely to churn than long-tenure customers
- Electronic check payers churn significantly more than any other payment method

## Project Structure

```
.
├── app.py                        # Streamlit dashboard
├── data/
│   ├── raw/                      # Telco-Customer-Churn.csv (not committed)
│   └── processed/                # X_train, X_test, y_train, y_test
├── models/                       # Trained model artifacts
│   ├── scaler.joblib
│   ├── logistic_regression.joblib
│   ├── xgboost.joblib
│   └── best_model.joblib
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── data/                     # validate.py, preprocess.py
│   ├── features/
│   ├── models/
│   └── visualization/
├── reports/                      # model_comparison.csv, roc_curve.png
├── api/                          # FastAPI scaffold
├── tests/
├── requirements.txt
└── ROADMAP.md
```


## Tech Stack

| Layer | Tools |
|-------|-------|
| Data | pandas, numpy |
| ML | scikit-learn, xgboost |
| Dashboard | Streamlit, Plotly |
| Serialisation | joblib |

## Dataset

**Telco-Customer-Churn.csv** — 7,043 customers, 21 features: tenure, contract type, internet service, payment method, monthly/total charges, and binary churn label.  
Class imbalance: ~26.5% churn.
