import pandas as pd
from sklearn.model_selection import train_test_split

# load raw data
df = pd.read_csv('data/raw/Telco-Customer-Churn.csv')

# drop customer ID (no signal)
df = df.drop(columns=['customerID'])

# fix TotalCharges: convert to number (blanks become NaN), then fill NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'])

# convert Yes/No columns to 1/0
yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
               'TechSupport', 'StreamingTV', 'StreamingMovies', 'Churn']

for col in yes_no_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# encode gender
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# encode Contract: Month-to-month=0, One year=1, Two year=2
df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})

# one-hot encode MultipleLines, InternetService, PaymentMethod
df = pd.get_dummies(df, columns=['MultipleLines', 'InternetService', 'PaymentMethod'])

# train/test split 80/20
X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# save to processed folder
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print('Train size:', X_train.shape)
print('Test size:', X_test.shape)
print('Columns:', list(X_train.columns))
print('Done. Files saved to data/processed/')
