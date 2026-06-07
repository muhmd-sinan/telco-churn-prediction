# Phase 3: Exploratory Data Analysis (EDA) - Power BI Implementation

**Objective**: Analyze churn patterns, identify indicators, generate business insights  
**Tool**: Power BI Desktop  
**Dataset**: `data/raw/Telco-Customer-Churn.csv`  
**Date**: 2026-06-07

---

## Step 1: Connect Power BI to Dataset

### Data Source Setup
1. Open Power BI Desktop
2. Home → Get Data → Text/CSV
3. Navigate to: `data/raw/netflix_user_behavior_dataset.csv`
4. Load dataset (50,000 rows × 20 columns)
5. Verify data types:
   - Numerical: age, account_age_months, monthly_fee, devices_used, avg_watch_time_minutes, watch_sessions_per_week, binge_watch_sessions, completion_rate, rating_given, content_interactions, recommendation_click_rate, days_since_last_login
   - Categorical: user_id, gender, country, subscription_type, payment_method, primary_device, favorite_genre, churned

### Data Type Corrections
- Right-click column → Change Type → Whole Number/Decimal Number/Text
- Ensure `churned` = Text (categorical)
- Ensure `monthly_fee` = Decimal Number
- Ensure `rating_given` = Decimal Number

---

## Step 2: Create Measures (DAX)

### Churn Metrics
```DAX
Total Customers = COUNTROWS('netflix_user_behavior_dataset')

Churned Customers = CALCULATE(COUNTROWS('netflix_user_behavior_dataset'), 'netflix_user_behavior_dataset'[churned] = "Yes")

Retained Customers = CALCULATE(COUNTROWS('netflix_user_behavior_dataset'), 'netflix_user_behavior_dataset'[churned] = "No")

Churn Rate = DIVIDE([Churned Customers], [Total Customers], 0)

Retention Rate = DIVIDE([Retained Customers], [Total Customers], 0)

Avg Age = AVERAGE('netflix_user_behavior_dataset'[age])

Avg Account Age = AVERAGE('netflix_user_behavior_dataset'[account_age_months])

Avg Monthly Fee = AVERAGE('netflix_user_behavior_dataset'[monthly_fee])

Avg Completion Rate = AVERAGE('netflix_user_behavior_dataset'[completion_rate])

Avg Watch Time = AVERAGE('netflix_user_behavior_dataset'[avg_watch_time_minutes])

Avg Days Since Login = AVERAGE('netflix_user_behavior_dataset'[days_since_last_login])
```

---

## Step 3: Create Visualizations

### Visualization 1: Churn Distribution
**Type**: Clustered Bar Chart  
**Purpose**: Show overall churn/retention ratio  
**Setup**:
- Axis: `churned` (Legend)
- Values: Count of user_id
- Format: Data labels ON, Title: "Customer Churn Distribution"

**Expected**: Retained ~80%, Churned ~20%

**Export**: Save as `reports/figures/churn_distribution.png`

---

### Visualization 2: Age vs Churn
**Type**: Histogram or Distribution Chart  
**Purpose**: Identify age groups with higher churn  
**Setup**:
- Axis: `age` (bin size: 5)
- Legend: `churned`
- Values: Count of user_id
- Add: Line chart with churn rate by age group

**Insight Expected**: Identify if certain age groups (18-25, 35-45, 55+) churn more

**Export**: Save as `reports/figures/age_vs_churn.png`

---

### Visualization 3: Subscription Type vs Churn
**Type**: Stacked Column Chart  
**Purpose**: Compare churn rates across subscription tiers  
**Setup**:
- Axis: `subscription_type`
- Legend: `churned`
- Values: Count of user_id
- Add: Churn rate measure as line

**Expected Finding**: Basic > Standard > Premium churn rates

**Export**: Save as `reports/figures/subscription_type_vs_churn.png`

---

### Visualization 4: Monthly Fee vs Churn
**Type**: Box Plot or Violin Plot (if available)  
**Alternative**: Scatter plot with `monthly_fee` (X) vs `churned` (color)  
**Setup**:
- Axis: `monthly_fee`
- Legend: `churned`
- Values: Count of user_id
- Add: Average monthly fee by churn status

**Insight**: Price sensitivity - higher fee correlates with churn?

**Export**: Save as `reports/figures/monthly_fee_vs_churn.png`

---

### Visualization 5: Engagement Metrics vs Churn (Multi-panel)
**Type**: Multi-row Card or Matrix  
**Purpose**: Compare engagement metrics between churned/retained  
**Metrics to Compare**:
- `completion_rate` (average)
- `recommendation_click_rate` (average)
- `content_interactions` (average)
- `avg_watch_time_minutes` (average)
- `rating_given` (average)
- `days_since_last_login` (average)

**Setup**:
- Matrix visual
- Rows: `churned`
- Values: Average of each metric
- Format: Conditional formatting (higher = red/green)

**Export**: Save as `reports/figures/engagement_metrics_vs_churn.png`

---

### Visualization 6: Country vs Churn
**Type**: Filled Map (Choropleth) or Clustered Bar Chart  
**Purpose**: Geographic churn distribution  
**Setup**:
- Location: `country`
- Color saturation: Churn Rate measure
- Format: Data labels ON

**Alternative**: Bar chart with top 10 countries by churn rate

**Export**: Save as `reports/figures/country_vs_churn.png`

---

### Visualization 7: Correlation Heatmap
**Power BI Limitation**: No native heatmap  
**Workaround**:
- Option A: Use Python visual in Power BI (requires matplotlib/seaborn)
- Option B: Create scatter plot matrix
- Option C: Export correlation table to CSV, visualize in separate tool

**Recommended**: Python visual in Power BI

**Python Script**:
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Select numerical columns
numerical_cols = ['age', 'account_age_months', 'monthly_fee', 'devices_used', 
                  'avg_watch_time_minutes', 'watch_sessions_per_week', 
                  'binge_watch_sessions', 'completion_rate', 'rating_given', 
                  'content_interactions', 'recommendation_click_rate', 
                  'days_since_last_login']

# Create correlation matrix
corr_matrix = dataset[numerical_cols].corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix - Numerical Features')
plt.tight_layout()
plt.savefig('reports/figures/correlation_heatmap.png', dpi=300)
```

**Export**: Save as `reports/figures/correlation_heatmap.png`

---

## Step 4: Statistical Testing (Optional in Power BI)

### Mann-Whitney U Test (DAX approximation)
Power BI cannot perform statistical tests natively. Options:
1. Use Python script visual for statistical tests
2. Calculate statistics in Python, display results in Power BI
3. Skip formal tests, rely on visual comparison

**Recommended**: Run statistical tests in separate Python script, document results.

---

## Step 5: Create Dashboard Layout

### Page 1: Executive Summary
- KPI cards: Total Customers, Churn Rate, Retention Rate, Avg Monthly Revenue
- Churn distribution bar chart
- Subscription type vs churn

### Page 2: Customer Demographics
- Age distribution
- Gender distribution
- Country distribution
- Device usage

### Page 3: Engagement Analysis
- Completion rate vs churn
- Watch time vs churn
- Days since login vs churn
- Recommendation click rate vs churn

### Page 4: Financial Analysis
- Monthly fee distribution
- Payment method analysis
- Subscription tier analysis

---

## Step 6: Export Visualizations

### Individual Visualizations
1. Click on visual → Export data (if needed)
2. File → Export → Export to PDF (full report)
3. Individual visuals: Copy visual → Paste to Paint/PowerPoint → Save as PNG

### Recommended Export Format
- **Full Dashboard**: `reports/netflix_churn_eda_dashboard.pbix`
- **PDF Export**: `reports/netflix_churn_eda_report.pdf`
- **Individual PNGs**: `reports/figures/*.png`

---

## Step 7: Document EDA Insights

After Power BI analysis complete, create: `reports/eda_insights.md`

### Template:
```markdown
# EDA Insights - Netflix Churn Prediction

## Key Findings

### 1. Churn Distribution
- Churn rate: X%
- Retained: X customers
- Churned: X customers

### 2. Top Churn Indicators
1. [Feature 1]: [Insight]
2. [Feature 2]: [Insight]
3. [Feature 3]: [Insight]

### 3. Demographic Insights
- Age groups most likely to churn: [X-Y years]
- Gender differences: [Insight]
- Geographic patterns: [Top countries]

### 4. Engagement Patterns
- Completion rate threshold: <X% = high churn risk
- Days since login threshold: >X days = critical
- Watch time correlation: [Insight]

### 5. Subscription Insights
- Highest churn: [Tier] subscription
- Price sensitivity: [Insight]
- Upgrade opportunities: [Recommendation]

### 6. Business Recommendations
1. [Action 1]
2. [Action 2]
3. [Action 3]

### 7. Next Steps
- Feature engineering: [Recommendations from EDA]
- Model features: [Top predictive features identified]
- Thresholds discovered: [Critical values for business rules]
```

---

## Deliverables Checklist
- [ ] Power BI file: `reports/netflix_churn_eda_dashboard.pbix`
- [ ] Churn distribution visualization
- [ ] Age vs churn visualization  
- [ ] Subscription type vs churn visualization
- [ ] Monthly fee vs churn visualization
- [ ] Engagement metrics comparison
- [ ] Country vs churn visualization
- [ ] Correlation heatmap
- [ ] EDA insights document: `reports/eda_insights.md`

---

## Alternative: R Script in Power BI
If Python visual not available, use R:
```r
library(ggplot2)
library(corrplot)

# Correlation matrix
numerical_cols <- dataset[, c('age', 'account_age_months', 'monthly_fee', ...)]
corr_matrix <- cor(numerical_cols)
corrplot(corr_matrix, method='color', type='upper')
```

---

## Support Files Needed
- Dataset: `data/raw/netflix_user_behavior_dataset.csv`
- Validation findings: `reports/data_validation_findings.md` (reference)

---

## Timeline
- Data connection: 5 min
- Measures creation: 10 min
- Visualizations: 30-45 min
- Dashboard layout: 15 min
- Export + documentation: 15 min
- **Total**: ~90 minutes

---

## Next Phase After EDA Complete
→ Phase 4: Feature Engineering (6 new features)
