# Netflix Churn Prediction - Project Memory Log

**Last Updated**: 2026-06-07  
**Project**: Customer Churn Prediction for Netflix-Style Streaming Platform  
**Status**: Phase 3 (EDA) In Progress - Waiting for Power BI completion

---

## Executive Summary

This file contains the complete project state from persistent memory storage, enabling seamless continuation of work across Claude Code sessions. The project is an end-to-end ML pipeline to predict customer churn 30 days in advance.

**Quick Status**:
- ✅ Phase 1 Complete (Foundation & Setup)
- ⏭️ Phase 2 Skipped (Data already clean)
- 🔄 Phase 3 In Progress (EDA - awaiting Power BI completion)
- ⏳ Phases 4-9 Pending (Feature Engineering through Deployment)

---

## Project Context

### Business Objectives
| Metric | Target | Status |
|--------|--------|--------|
| Recall | >80% | Not started |
| ROC-AUC | >85% | Not started |
| Churn Reduction | 20% | Not started |
| Expected ROI | $100K net gain (3.3x) | Projected |

### Timeline
- **Total Duration**: 10 days
- **Completed**: Day 1 (Phase 1)
- **Skipped**: Day 2 (Phase 2 - data clean, not needed)
- **Current**: Day 3 (Phase 3 - EDA)
- **Remaining**: Days 4-10 (Phases 4-9)

### Dataset Summary
- **File**: `data/raw/netflix_user_behavior_dataset.csv`
- **Shape**: 50,000 rows × 20 columns
- **Target**: `churned` (19.93% churn rate - 9,964 Yes / 40,036 No)
- **Features**: 12 numerical, 8 categorical
- **Quality**: ✅ Perfect - 0 missing values, 0 duplicates, 0 outliers

---

## Completed Work (Phase 1)

### 1. Environment Setup ✅
```powershell
# Virtual environment
.\venv\Scripts\Activate.ps1

# Key dependencies installed:
# - pandas, numpy, matplotlib, seaborn, scikit-learn
# - xgboost, lightgbm, catboost, imbalanced-learn, shap
# - jupyter, fastapi, uvicorn, joblib
```

**Files Created**:
- `venv/` - Virtual environment (active)
- `requirements.txt` - Frozen dependencies

### 2. Project Structure ✅
```
netflix_churn_prediction/
├── data/
│   ├── raw/
│   │   └── netflix_user_behavior_dataset.csv  ✅ Validated
│   ├── processed/                             ✅ Created (empty)
│   └── predictions/                           ✅ Created (empty)
├── src/
│   └── data/
│       └── data_validator.py                  ✅ Working
├── reports/
│   ├── figures/                               ✅ Created (empty)
│   ├── data_validation_findings.md            ✅ Complete
│   └── phase3_eda_powerbi_guide.md            ✅ Complete
├── requirements.txt                           ✅ Complete
├── ROADMAP.md                                 ✅ Reference
└── README.md                                  ❌ Not yet created
```

### 3. Data Validation ✅
**Script**: `src/data/data_validator.py`

**Key Findings**:
| Check | Result | Details |
|-------|--------|---------|
| Shape | ✅ PASS | 50,000 × 20 (as expected) |
| Missing Values | ✅ PASS | 0 total |
| Duplicates | ✅ PASS | 0 exact, 0 user_id duplicates |
| Outliers | ✅ PASS | 0 detected (IQR method) |
| Churn Rate | ✅ PASS | 19.93% (within 15-25% expected) |
| Business Rules | ✅ PASS | All 6 constraints passed |

**Feature Classification**:
```python
# Identifier (remove for modeling)
- user_id

# Demographic (5) - One-Hot encoding
- age, gender, country

# Subscription (5) - Label encoding for subscription_type
- subscription_type, monthly_fee, payment_method, account_age_months

# Behavioral (4) - Scale with StandardScaler
- avg_watch_time_minutes, watch_sessions_per_week, binge_watch_sessions, completion_rate

# Engagement (6) - Scale with StandardScaler
- rating_given, content_interactions, recommendation_click_rate, primary_device, devices_used, favorite_genre

# Risk Indicator (1) - Key predictor
- days_since_last_login

# Target
- churned (Yes/No)
```

**Encoding Strategy**:
- **One-Hot**: gender, country, payment_method, favorite_genre, primary_device
- **Label/Ordinal**: subscription_type (Basic=0, Standard=1, Premium=2)
- **Scale**: All 12 numerical features (StandardScaler)
- **Remove**: user_id

### 4. Phase 2 Decision - SKIPPED ⏭️
**Reason**: Data validation passed all checks with zero issues
- No missing values → No imputation needed
- No duplicates → No deduplication needed
- No outliers → No outlier handling needed
- All business rules valid → No data cleaning needed

**Impact**: Saves 1 day, accelerates to Phase 3

### 5. Phase 3 Setup - Power BI EDA ✅
**Decision**: Use Power BI instead of Python/Jupyter notebooks

**Rationale**:
- Interactive dashboards better for stakeholder presentations
- Drag-and-drop interface faster than coding each visualization
- Easy export to PDF/PPT for business reports

**Created**: `reports/phase3_eda_powerbi_guide.md`
- 7 required visualizations documented
- DAX measures for churn metrics
- Dashboard layout (4 pages)
- Statistical testing alternatives

---

## Current State (Phase 3 - In Progress)

### What User Needs to Complete
1. Create Power BI dashboard: `reports/netflix_churn_eda_dashboard.pbix`
2. Generate 7 visualizations:
   - Churn distribution (bar chart)
   - Age vs churn (histogram + KDE)
   - Subscription type vs churn (grouped bar)
   - Monthly fee vs churn (box plot)
   - Engagement metrics vs churn (multi-panel)
   - Country vs churn (choropleth/bar)
   - Correlation heatmap (matrix)
3. Export visualizations to `reports/figures/*.png`
4. Document insights in `reports/eda_insights.md`

### Expected Insights to Validate
1. **Days since last login**: Strong positive correlation with churn (>14 days = high risk)
2. **Completion rate**: Strong negative correlation with churn (<40% = critical)
3. **Account age**: New customers (<6 months) higher churn risk
4. **Subscription type**: Basic tier highest churn rate
5. **Monthly fee**: Premium pricing correlates with higher churn (price sensitivity)
6. **Engagement metrics**: Low engagement = high churn risk

### Statistical Tests to Document
- **Mann-Whitney U Test**: For continuous features (engagement metrics vs churn)
- **Chi-Square Test**: For categorical features (subscription type, gender vs churn)

---

## Upcoming Work (Phases 4-9)

### Phase 4: Feature Engineering (Day 4) ⏳
**Create 6 engineered features**:

| Feature | Formula | Range | Business Meaning |
|---------|---------|-------|------------------|
| engagement_score | completion_rate×0.3 + recommendation_click_rate×0.3 + content_interactions_norm×0.2 + rating_given_norm×0.2 | [0,1] | Overall platform interaction quality |
| activity_score | watch_sessions_per_week×0.4 + avg_watch_time_norm×0.3 + binge_sessions_norm×0.3 | Continuous | Viewing frequency & intensity |
| binge_ratio | binge_watch_sessions / watch_sessions_per_week | [0,1] | Content affinity indicator |
| login_risk | pd.cut(days_since_last_login, bins=[-1,7,14,30,∞]) | low/medium/high/critical | Risk segmentation |
| is_premium | subscription_type == 'Premium' | {0,1} | Premium user indicator |
| content_affinity_score | mean engagement per favorite_genre | [0,1] | Genre loyalty indicator |

**Expected Correlations with Churn**:
- engagement_score: -0.35 to -0.45 (strong negative)
- activity_score: -0.20 to -0.30 (moderate negative)
- login_risk_high/critical: +0.50 to +0.60 (strong positive)
- is_premium: -0.15 to -0.25 (moderate negative)

### Phase 5: Data Preprocessing (Day 5) ⏳
**Tasks**:
- Apply One-Hot Encoding to nominal features
- Apply Label Encoding to ordinal features (subscription_type, login_risk)
- Scale numerical features with StandardScaler
- Split dataset (80/20) with stratification
- Handle class imbalance with SMOTE + Tomek Links

**Pipeline Output**:
- `data/processed/train_features.csv`
- `data/processed/test_features.csv`
- `models/preprocessor.pkl` (saved pipeline)

### Phase 6: Model Development (Days 6-7) ⏳
**Train 6 models**:
1. Logistic Regression (baseline)
2. Decision Tree
3. Random Forest
4. **XGBoost** (RECOMMENDED - champion model)
5. LightGBM
6. CatBoost

**Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation

**Expected Performance**:
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.78 | 0.62 | 0.81 | 0.70 | 0.85 |
| Decision Tree | 0.82 | 0.68 | 0.76 | 0.72 | 0.82 |
| Random Forest | 0.86 | 0.75 | 0.84 | 0.79 | 0.91 |
| **XGBoost** | **0.87** | **0.76** | **0.86** | **0.81** | **0.93** |
| LightGBM | 0.86 | 0.74 | 0.85 | 0.79 | 0.92 |
| CatBoost | 0.87 | 0.77 | 0.85 | 0.81 | 0.93 |

### Phase 7: Model Evaluation (Day 8) ⏳
**Metrics**:
- Primary: Recall (>80%), ROC-AUC (>85%)
- Secondary: Precision, F1, Accuracy
- Visualizations: Confusion matrix, ROC curves, Precision-Recall curves

**Threshold Optimization**: Target Recall = 85%

### Phase 8: Feature Importance & SHAP (Day 8) ⏳
**Expected Top 5 Features**:
1. days_since_last_login (28%)
2. completion_rate (18%)
3. engagement_score (15%)
4. account_age_months (12%)
5. monthly_fee (8%)

**Deliverables**:
- SHAP summary plots
- Feature importance ranking
- Individual prediction explanations
- Business action recommendations

### Phase 9: Customer Segmentation (Day 9) ⏳
**Risk Segments**:
| Segment | Size | Churn Prob | Budget/Customer | Actions |
|---------|------|------------|-----------------|---------|
| Low Risk | 45% | <30% | $2/month | Loyalty rewards, referrals |
| Medium Risk | 35% | 30-70% | $8/month | Personalized content, minor discounts |
| High Risk | 20% | >70% | $15/month | Aggressive offers, account manager |

### Phase 10: Deployment (Day 10) ⏳
**Deliverables**:
- Batch prediction pipeline
- Real-time API (FastAPI)
- Docker containerization
- Monitoring dashboard

---

## Domain Knowledge - Churn Drivers

### High Impact Predictors
- **days_since_last_login**: Strongest predictor (>14 days = high risk, >30 days = critical)

### Medium Impact Predictors
- completion_rate
- account_age_months
- monthly_fee
- recommendation_click_rate
- subscription_type

### Low Impact (Exploratory)
- age, gender, country, favorite_genre, payment_method

### Hypothesis
Low completion (<40%), new customers (<6 months), premium pricing, and poor personalization correlate with higher churn.

---

## Key Technical Decisions

### 1. Class Imbalance Strategy
- **Finding**: 19.93% churn rate (80:20 ratio)
- **Plan**: SMOTE + Tomek Links in Phase 5
- **Alternative**: scale_pos_weight in XGBoost, class_weight='balanced' in sklearn

### 2. Model Selection
- **Champion**: XGBoost (highest ROC-AUC + Recall, fast inference)
- **Alternative**: CatBoost (minimal preprocessing)
- **Fallback**: Random Forest (interpretability)

### 3. Evaluation Focus
- **Primary**: Recall >80% (catch at-risk customers)
- **Secondary**: ROC-AUC >85% (ranking quality)
- **Why**: Missing a churner = lost revenue

### 4. Encoding Strategy
- Tree models (XGBoost, Random Forest): No scaling needed
- Linear models: StandardScaler required
- Categoricals: One-Hot for nominal, Label for ordinal

---

## Files & Locations

### Critical Files
| File | Purpose | Status |
|------|---------|--------|
| `data/raw/netflix_user_behavior_dataset.csv` | Raw dataset | ✅ Validated |
| `src/data/data_validator.py` | Validation script | ✅ Working |
| `reports/data_validation_findings.md` | Phase 1 results | ✅ Complete |
| `reports/phase3_eda_powerbi_guide.md` | EDA instructions | ✅ Complete |
| `ROADMAP.md` | Full project plan | ✅ Reference |
| `PROJECT_MEMORY.md` | This file | ✅ Created |

### Pending Files
| File | Expected Phase | Purpose |
|------|----------------|---------|
| `reports/netflix_churn_eda_dashboard.pbix` | Phase 3 | Power BI dashboard |
| `reports/figures/*.png` | Phase 3 | Visualization exports |
| `reports/eda_insights.md` | Phase 3 | EDA findings |
| `data/processed/train_features.csv` | Phase 4 | Engineered features |
| `notebooks/02_Feature_Engineering.ipynb` | Phase 4 | Feature engineering |
| `notebooks/03_Model_Training.ipynb` | Phase 6 | Model training |
| `models/saved_models/xgboost_churn_model.pkl` | Phase 6 | Trained model |
| `reports/feature_importance_business.md` | Phase 8 | SHAP insights |
| `reports/customer_segmentation.md` | Phase 9 | Risk segments |

---

## Commands Reference

### Activate Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### Run Validation
```powershell
python src/data/data_validator.py
```

### Check Dataset
```powershell
python -c "import pandas as pd; df = pd.read_csv('data/raw/netflix_user_behavior_dataset.csv'); print(df.shape); print(df['churned'].value_counts())"
```

### Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## Next Steps for Continuation

### Option 1: Check Phase 3 Progress
1. Verify `reports/netflix_churn_eda_dashboard.pbix` exists
2. Check `reports/figures/` for 7 PNG visualizations
3. Review `reports/eda_insights.md` for documented findings
4. If complete → proceed to Phase 4
5. If incomplete → continue Power BI work

### Option 2: Resume at Phase 4 (Feature Engineering)
If Phase 3 is complete, next tasks:
1. Load validated dataset
2. Create 6 engineered features (engagement_score, activity_score, etc.)
3. Validate correlations with churn
4. Save to `data/processed/train_features.csv`
5. Store feature definitions in memory

### Option 3: Fast-Forward to Modeling
If EDA already understood:
1. Skip Phase 3 completion check
2. Create minimal feature engineering
3. Proceed to Phase 5 (Preprocessing)
4. Phase 6 (Model Training) with XGBoost

---

## Persistent Memory Entries

This project has the following entries in AgentDB memory:

1. **project-roadmap** (netflix-churn:project-context)
   - Timeline, deliverables, success metrics, ROI projection

2. **roadmap-created** (netflix-churn:episodic)
   - File location, sections, status

3. **project-overview** (netflix-churn:project-context)
   - Dataset info, goal, success metrics

4. **feature-classification** (netflix-churn:project-context)
   - Feature categories: identifier, demographic, subscription, behavioral, engagement, risk_indicator, target

5. **domain-knowledge-churn-drivers** (netflix-churn:insights)
   - High/medium/low impact predictors, hypothesis

---

## Contact & Context

**Project Root**: `C:\Users\m21si\OneDrive\Desktop\Zoople\Netflix Churn Prediction`

**Git Status**:
- Commits: 4 total
- Latest: "roadmad decided"
- Branches: main

**Session Notes**:
- User prefers Power BI over Jupyter notebooks for EDA
- User requested persistent memory log (this file)
- Phase 2 skipped due to clean data
- Phase 3 waiting for user Power BI work

---

## Summary

**What We've Done**:
- Set up complete Python ML environment
- Validated 50K-row dataset (perfect quality)
- Created project structure
- Documented all validation findings
- Created Power BI EDA guide
- Skipped Phase 2 (data already clean)

**What We're Doing**:
- Phase 3 EDA: User creating Power BI dashboard
- Waiting for: 7 visualizations + insights document

**What's Next**:
- Phase 4: Feature Engineering (6 new features)
- Phase 5: Preprocessing (encoding, scaling, SMOTE)
- Phase 6: Model Training (6 models, XGBoost champion)
- Phase 7-9: Evaluation, SHAP, Segmentation
- Phase 10: Deployment

---

*This file serves as the single source of truth for project state. Update after each session to maintain continuity.*
