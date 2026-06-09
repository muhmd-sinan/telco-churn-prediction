import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve


# --------------------------------------------------
# Step 1: Load the processed test data
# --------------------------------------------------

X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

X_test = X_test.fillna(0)


# --------------------------------------------------
# Step 2: Load the saved scaler and models
# (these were saved by train.py — no retraining here)
# --------------------------------------------------

scaler = joblib.load('models/scaler.joblib')
logistic_model = joblib.load('models/logistic_regression.joblib')
random_forest_model = joblib.load('models/random_forest.joblib')
xgboost_model = joblib.load('models/xgboost.joblib')

print('Models loaded successfully.')
print()


# --------------------------------------------------
# Step 3: Scale the test data using the saved scaler
# --------------------------------------------------

X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# Step 4: Get predictions from each model
# predict_proba gives churn probability (for AUC and ROC curve)
# predict gives hard 0/1 labels (for confusion matrix and report)
# --------------------------------------------------

logistic_proba = logistic_model.predict_proba(X_test_scaled)[:, 1]
logistic_labels = logistic_model.predict(X_test_scaled)

random_forest_proba = random_forest_model.predict_proba(X_test_scaled)[:, 1]
random_forest_labels = random_forest_model.predict(X_test_scaled)

xgboost_proba = xgboost_model.predict_proba(X_test_scaled)[:, 1]
xgboost_labels = xgboost_model.predict(X_test_scaled)


# --------------------------------------------------
# Step 5: Print confusion matrix and classification report
# Confusion matrix format:
#   [[True Negative,  False Positive],
#    [False Negative, True Positive ]]
# --------------------------------------------------

print('===== Logistic Regression =====')
print('Confusion Matrix:')
print(confusion_matrix(y_test, logistic_labels))
print()
print('Classification Report:')
print(classification_report(y_test, logistic_labels, target_names=['No Churn', 'Churn']))

print('===== Random Forest =====')
print('Confusion Matrix:')
print(confusion_matrix(y_test, random_forest_labels))
print()
print('Classification Report:')
print(classification_report(y_test, random_forest_labels, target_names=['No Churn', 'Churn']))

print('===== XGBoost =====')
print('Confusion Matrix:')
print(confusion_matrix(y_test, xgboost_labels))
print()
print('Classification Report:')
print(classification_report(y_test, xgboost_labels, target_names=['No Churn', 'Churn']))


# --------------------------------------------------
# Step 6: Calculate AUC-ROC scores
# --------------------------------------------------

logistic_auc = roc_auc_score(y_test, logistic_proba)
random_forest_auc = roc_auc_score(y_test, random_forest_proba)
xgboost_auc = roc_auc_score(y_test, xgboost_proba)

print('===== AUC-ROC Scores =====')
print('Logistic Regression :', round(logistic_auc, 4))
print('Random Forest       :', round(random_forest_auc, 4))
print('XGBoost             :', round(xgboost_auc, 4))
print()


# --------------------------------------------------
# Step 7: Plot ROC curves for all three models
# --------------------------------------------------

logistic_fpr, logistic_tpr, logistic_thresholds = roc_curve(y_test, logistic_proba)
random_forest_fpr, random_forest_tpr, random_forest_thresholds = roc_curve(y_test, random_forest_proba)
xgboost_fpr, xgboost_tpr, xgboost_thresholds = roc_curve(y_test, xgboost_proba)

plt.figure(figsize=(8, 6))

plt.plot(logistic_fpr, logistic_tpr, label='Logistic Regression (AUC = ' + str(round(logistic_auc, 4)) + ')')
plt.plot(random_forest_fpr, random_forest_tpr, label='Random Forest (AUC = ' + str(round(random_forest_auc, 4)) + ')')
plt.plot(xgboost_fpr, xgboost_tpr, label='XGBoost (AUC = ' + str(round(xgboost_auc, 4)) + ')')

plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — All Models')
plt.legend()
plt.tight_layout()
plt.savefig('reports/roc_curve.png')
plt.show()

print('ROC curve saved to reports/roc_curve.png')
print('Done.')
