import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Step 1: Load the processed data
# --------------------------------------------------

X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# some service columns have NaN from "No internet service" / "No phone service" values
# that were not handled in preprocessing — fill them with 0 (meaning: doesn't have the service)
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print()


# --------------------------------------------------
# Step 2: Scale the features
# Logistic Regression needs scaled input to work well.
# Random Forest and XGBoost don't need it, but it
# doesn't hurt them either, so we scale for all three.
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# Step 3: Train Logistic Regression
# --------------------------------------------------

print('Training Logistic Regression...')

logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train_scaled, y_train)

logistic_predictions = logistic_model.predict_proba(X_test_scaled)[:, 1]
logistic_auc = roc_auc_score(y_test, logistic_predictions)

print('Logistic Regression AUC-ROC:', round(logistic_auc, 4))
print()


# --------------------------------------------------
# Step 4: Train Random Forest
# --------------------------------------------------

print('Training Random Forest...')

random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train_scaled, y_train)

random_forest_predictions = random_forest_model.predict_proba(X_test_scaled)[:, 1]
random_forest_auc = roc_auc_score(y_test, random_forest_predictions)

print('Random Forest AUC-ROC:', round(random_forest_auc, 4))
print()


# --------------------------------------------------
# Step 5: Train XGBoost
# --------------------------------------------------

print('Training XGBoost...')

xgboost_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0)
xgboost_model.fit(X_train_scaled, y_train)

xgboost_predictions = xgboost_model.predict_proba(X_test_scaled)[:, 1]
xgboost_auc = roc_auc_score(y_test, xgboost_predictions)

print('XGBoost AUC-ROC:', round(xgboost_auc, 4))
print()


# --------------------------------------------------
# Step 6: Compare all three models
# --------------------------------------------------

print('------ Model Comparison ------')
print('Logistic Regression  AUC-ROC:', round(logistic_auc, 4))
print('Random Forest        AUC-ROC:', round(random_forest_auc, 4))
print('XGBoost              AUC-ROC:', round(xgboost_auc, 4))
print()


# --------------------------------------------------
# Step 7: Find the best model by AUC-ROC score
# --------------------------------------------------

all_scores = {
    'Logistic Regression': logistic_auc,
    'Random Forest': random_forest_auc,
    'XGBoost': xgboost_auc
}

best_model_name = max(all_scores, key=all_scores.get)
best_model_score = all_scores[best_model_name]

print('Best model:', best_model_name)
print('Best AUC-ROC:', round(best_model_score, 4))


# --------------------------------------------------
# Step 8: Save the comparison to a CSV report
# --------------------------------------------------

results_data = {
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'AUC_ROC': [round(logistic_auc, 4), round(random_forest_auc, 4), round(xgboost_auc, 4)]
}

results_df = pd.DataFrame(results_data)
results_df = results_df.sort_values('AUC_ROC', ascending=False)
results_df = results_df.reset_index(drop=True)

results_df.to_csv('reports/model_comparison.csv', index=False)

print()
print('Results saved to reports/model_comparison.csv')


# --------------------------------------------------
# Step 9: Save all models and the scaler to disk
# --------------------------------------------------

joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(logistic_model, 'models/logistic_regression.joblib')
joblib.dump(random_forest_model, 'models/random_forest.joblib')
joblib.dump(xgboost_model, 'models/xgboost.joblib')

print('Scaler saved to models/scaler.joblib')
print('Logistic Regression saved to models/logistic_regression.joblib')
print('Random Forest saved to models/random_forest.joblib')
print('XGBoost saved to models/xgboost.joblib')
print('Done.')
