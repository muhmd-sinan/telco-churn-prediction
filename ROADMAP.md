# Churn Prediction Model — Project Roadmap

**Dataset:** Telco-Customer-Churn.csv (7,043 rows × 21 cols)  
**Target:** `Churn` (26.5% positive — class imbalance)  
**Data note:** 11 blank `TotalCharges` (new customers, tenure≈0)

---

## Tech Stack

```
Python 3.11+
pandas / numpy                 — data
matplotlib / seaborn / plotly  — viz
scikit-learn                   — pipeline, models, metrics
xgboost / lightgbm             — gradient boosting
shap                           — explainability
streamlit                      — dashboard
fastapi + joblib               — serving (optional)
pytest                         — testing
```

---

## Project Structure

```
project/
├── data/
│   ├── raw/                  ← Telco-Customer-Churn.csv (read-only)
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data/                 ← ingest.py, validate.py
│   ├── features/             ← build_features.py
│   ├── models/               ← train.py, evaluate.py, predict.py
│   └── visualization/        ← plots.py
├── tests/
├── models/                   ← serialized artifacts
├── reports/
└── config.yaml
```

---

## Phase 1 — Foundation (Day 1)

**Goal:** Reproducible project skeleton, raw data locked.

- [ ] Init git repo, `.gitignore` (ignore `.env`, model artifacts >50MB)
- [ ] Create `venv`, pin `requirements.txt`
- [ ] Build directory structure above
- [ ] Write `src/data/validate.py` — schema check on load (col names, dtypes, row count)
- [ ] `config.yaml` — paths, random seed, test split ratio

**Deliverable:** Clean repo, data validated, environment reproducible.

---

## Phase 2 — Exploratory Data Analysis (Day 2–3)

**Goal:** Understand distributions, find signal, expose data quality issues.

- [ ] Univariate: distribution of all 20 features, note 26.5% imbalance
- [ ] Bivariate vs `Churn`:
  - `tenure` — churners have lower tenure (expect strong signal)
  - `Contract` — Month-to-month churns far more
  - `MonthlyCharges` — higher charges → higher churn
  - `InternetService` (Fiber optic) — suspected high churn
  - `PaymentMethod` (Electronic check) — suspected correlation
- [ ] Missing values: 11 blank `TotalCharges`
- [ ] Correlation matrix: numerical features
- [ ] Churn rate by segment: Contract × tenure bucket × PaymentMethod
- [ ] Generate HTML EDA report

**Key questions:**
1. Which contract type has highest churn? (hypothesis: Month-to-month)
2. Does adding services reduce churn? (loyalty effect)
3. Is there a tenure "danger zone" (months 1–12)?
4. Does MonthlyCharges alone predict churn?

**Deliverable:** `notebooks/01_eda.ipynb` + `reports/eda_report.html`

---

## Phase 3 — Data Preprocessing (Day 4)

**Goal:** Clean, encode, scale → reproducible sklearn Pipeline.

- [ ] Drop `customerID` (no signal)
- [ ] Fix `TotalCharges`: 11 blanks → impute as `tenure × MonthlyCharges`, cast to `float64`
- [ ] Binary encode (Yes→1, No→0): `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`, all service cols
- [ ] Ordinal encode `Contract`: Month-to-month=0, One year=1, Two year=2
- [ ] One-hot encode: `gender`, `MultipleLines`, `InternetService`, `PaymentMethod`
- [ ] Scale numericals (StandardScaler): `tenure`, `MonthlyCharges`, `TotalCharges`
- [ ] Stratified train/test split: 80/20, `random_state=42`
- [ ] Wrap as `sklearn.Pipeline` for leak-free transforms

**Deliverable:** `src/data/preprocess.py`, fitted pipeline saved to `models/preprocessor.joblib`

---

## Phase 4 — Feature Engineering (Day 4–5)

**Goal:** Domain-informed features that boost model signal.

| Feature | Formula | Rationale |
|---|---|---|
| `service_count` | sum of 8 service cols | Loyalty proxy — more services = harder to leave |
| `monthly_to_total_ratio` | `MonthlyCharges / TotalCharges` | High ratio = recent customer |
| `tenure_bucket` | 0-12, 13-24, 25-48, 49-72 | Non-linear tenure risk |
| `has_support_services` | `OnlineSecurity OR TechSupport` | Engagement signal |
| `charge_per_service` | `MonthlyCharges / (service_count+1)` | Value perception |

- [ ] Implement all features above
- [ ] Validate no leakage (all computable from raw features)

**Deliverable:** `src/features/build_features.py`

---

## Phase 5 — Model Development (Day 5–7)

**Goal:** Train, tune, compare 5+ models. Handle class imbalance.

**Imbalance strategy:** `class_weight='balanced'` or `scale_pos_weight` (XGBoost). Stratified k-fold throughout. Keep SMOTE inside Pipeline if used to prevent leakage.

**Models:**
- [ ] Logistic Regression — interpretable baseline
- [ ] Decision Tree — understand splits
- [ ] Random Forest — robust ensemble baseline
- [ ] XGBoost — primary gradient boost candidate
- [ ] LightGBM — faster alternative
- [ ] Stacking Ensemble — LR meta-learner over RF + XGB + LGB

**Tuning:**
- [ ] RandomizedSearchCV for initial sweep
- [ ] Optuna (optional) for final XGBoost/LGB tuning
- [ ] 5-fold stratified CV throughout

**Primary metric:** AUC-ROC  
**Secondary metric:** Recall (missing a churner costs more than false alarm)

**Deliverable:** `src/models/train.py`, `notebooks/03_modeling.ipynb`, all models in `models/`

---

## Phase 6 — Evaluation & Explainability (Day 7–8)

**Goal:** Rigorous evaluation + business-interpretable explanations.

- [ ] Confusion matrix per model
- [ ] ROC curves (all models on one plot)
- [ ] Precision-Recall curves
- [ ] Threshold optimization: maximize F1 or business cost metric
- [ ] Business cost matrix: FN cost (~$500 lost customer) vs FP cost (~$20 retention offer)
- [ ] SHAP values:
  - Global: feature importance bar + beeswarm
  - Local: waterfall plot for individual high-risk customers
  - Interaction: `tenure` × `Contract`

**Expected top features (hypothesis):**
1. `Contract` type
2. `tenure`
3. `MonthlyCharges`
4. `InternetService` (Fiber)
5. `TechSupport` / `OnlineSecurity`

**Deliverable:** `src/models/evaluate.py`, `reports/model_evaluation.html`

---

## Phase 7 — Insights Dashboard (Day 8–9)

**Goal:** Stakeholder-facing Streamlit app.

**Tabs:**
- [ ] **Overview** — churn rate KPI, at-risk customer count, revenue at risk
- [ ] **EDA** — interactive Plotly charts, filter by segment
- [ ] **Model Performance** — ROC/PR curves, confusion matrix, metric table
- [ ] **Explainability** — global SHAP importance + individual prediction explainer
- [ ] **Predict** — upload CSV → batch score → download with `churn_probability`, `risk_tier`

**Deliverable:** `app.py` → `streamlit run app.py`

---

## Phase 8 — Production Pipeline (Day 9–10)

**Goal:** Reproducible scoring. Optional API.

- [ ] `src/models/predict.py` — batch scoring (CSV in → CSV out with `churn_probability`, `churn_prediction`, `risk_tier`)
- [ ] `models/best_model.joblib` — final serialized pipeline (preprocessor + model)
- [ ] FastAPI endpoint `api/main.py` — `/predict` POST, single or batch
- [ ] Unit tests:
  - `tests/test_preprocess.py` — pipeline transforms
  - `tests/test_features.py` — feature engineering
  - `tests/test_predict.py` — output shape + range
- [ ] `README.md` — setup, usage, results summary

**Deliverable:** Batch scorer + API + test suite

---

## Timeline

| Phase | Days | Deliverable |
|---|---|---|
| 1 — Foundation | 1 | Repo, structure, validation |
| 2 — EDA | 2–3 | EDA notebook + report |
| 3 — Preprocessing | 4 | sklearn Pipeline |
| 4 — Feature Engineering | 4–5 | Feature builder |
| 5 — Modeling | 5–7 | 5 trained models |
| 6 — Evaluation | 7–8 | Metrics + SHAP |
| 7 — Dashboard | 8–9 | Streamlit app |
| 8 — Production | 9–10 | Batch scorer + API + tests |

**Total:** ~10 days solo, ~5–6 days with parallel agents.

---

## Key Risks

| Risk | Mitigation |
|---|---|
| Class imbalance inflates accuracy | Use AUC-ROC + Recall, not accuracy |
| Data leakage in CV | All transforms inside Pipeline |
| `TotalCharges` blank causes crash | Impute before pipeline fit |
| Overfitting on 7k rows | CV + regularization + early stopping |
| SHAP slow on ensemble | `TreeExplainer`, sample 500 rows for viz |
