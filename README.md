#  Customer Churn Prediction

A machine learning project that predicts whether a telecom customer will churn,
built with Logistic Regression and deployed as a REST API using FastAPI.

##  Problem Statement
Telecom companies lose revenue when customers leave. This model predicts churn
probability using customer data, enabling proactive retention strategies.

##  Project Structure
```
customer-churn-prediction/
├── app/
│   └── main.py          # FastAPI application
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   ├── churn_model.pkl  # Trained model
│   └── scaler.pkl       # Feature scaler
├── model.py             # Training script
├── requirements.txt
└── README.md
```

##  Models Compared
| Model | Accuracy | Precision | Recall | F1-Score |

| Logistic Regression | 0.805 | 0.655 | 0.559 | 0.603 |
| Random Forest | 0.786 | 0.625 | 0.487 | 0.547 |
| XGBoost | 0.786 | 0.609 | 0.537 | 0.571 |

 **Logistic Regression selected** as final model (best F1-Score)

##  Key Findings (SHAP Analysis)
Top features driving churn:
1. tenure — New customers churn more
2. Contract_Two_year — Long term contracts reduce churn
3. MonthlyCharges — Higher charges increase churn risk

##  Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Suhail00025/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python model.py
```

### 4. Run the API
```bash
uvicorn app.main:app --reload
```

### 5. Test the API
Open browser and go to:
```
http://127.0.0.1:8000/docs
```

##  API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Predict customer churn |

### Sample Request
```json
{
  "gender": 1,
  "SeniorCitizen": 0,
  "Partner": 1,
  "tenure": 12,
  "MonthlyCharges": 65.5,
  "TotalCharges": 786.0,
  "Contract_Two_year": 0,
  "PaymentMethod_Electronic_check": 1
}
```

### Sample Response
```json
{
  "churn_prediction": 1,
  "churn_label": "Yes",
  "churn_probability": 0.792,
  "message": "This customer is likely to churn!"
}
```

##  Improvements To Make
- Handle class imbalance with SMOTE
- Hyperparameter tuning with GridSearchCV
- Feature engineering (charges per tenure)
- K-fold cross validation
- Deploy on cloud (Render/AWS)

##  Tech Stack
- Python, Pandas, Scikit-learn, XGBoost
- SHAP for model explainability
- FastAPI + Uvicorn for API
- GitHub for version control