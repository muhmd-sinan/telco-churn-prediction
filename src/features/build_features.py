import pandas as pd

X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')


def add_features(df):
    # count of services the customer has
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies',
                    'PhoneService', 'MultipleLines_Yes']
    df['service_count'] = df[service_cols].sum(axis=1)

    # ratio of monthly to total charges (high = new customer)
    df['monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)

    # tenure bucket: 0-12, 13-24, 25-48, 49-72
    bins = [0, 12, 24, 48, 72]
    labels = [0, 1, 2, 3]
    df['tenure_bucket'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True).astype(int)

    # has support services: OnlineSecurity or TechSupport
    df['has_support_services'] = ((df['OnlineSecurity'] == 1) | (df['TechSupport'] == 1)).astype(int)

    # charge per service
    df['charge_per_service'] = df['MonthlyCharges'] / (df['service_count'] + 1)

    return df


X_train = add_features(X_train)
X_test = add_features(X_test)

X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)

print('New columns added:', ['service_count', 'monthly_to_total_ratio', 'tenure_bucket', 'has_support_services', 'charge_per_service'])
print('Train shape:', X_train.shape)
print('Done.')
