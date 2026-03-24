import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df.drop(columns=['customerID'], inplace=True)

# Encode
binary_cols = [col for col in df.columns 
               if df[col].nunique() == 2 and df[col].dtype == 'object']
multi_cols = [col for col in df.columns 
              if df[col].nunique() > 2 and df[col].dtype == 'object']

le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

df = pd.get_dummies(df, columns=multi_cols, drop_first=True)


df.columns = df.columns.str.replace(' ', '_')

print("Columns after encoding:")
print(df.columns.tolist())

# Scale
scaler = StandardScaler()
scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# Train
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Save
joblib.dump(model, 'models/churn_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("\n Model saved successfully!")
print("Features used:", X.columns.tolist())

print("Exact column names:")
for col in X.columns.tolist():
    print(f'  "{col}"')