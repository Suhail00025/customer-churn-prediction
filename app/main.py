from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

app = FastAPI(title="Customer Churn Prediction API")

# Define input schema 
class CustomerData(BaseModel):
    gender: int                                       
    SeniorCitizen: int                                 
    Partner: int                                       
    Dependents: int                                   
    tenure: float                                     
    PhoneService: int                                  
    PaperlessBilling: int                              
    MonthlyCharges: float
    TotalCharges: float
    MultipleLines_No_phone_service: int = 0
    MultipleLines_Yes: int = 0
    InternetService_Fiber_optic: int = 0
    InternetService_No: int = 0
    OnlineSecurity_No_internet_service: int = 0
    OnlineSecurity_Yes: int = 0
    OnlineBackup_No_internet_service: int = 0
    OnlineBackup_Yes: int = 0
    DeviceProtection_No_internet_service: int = 0
    DeviceProtection_Yes: int = 0
    TechSupport_No_internet_service: int = 0
    TechSupport_Yes: int = 0
    StreamingTV_No_internet_service: int = 0
    StreamingTV_Yes: int = 0
    StreamingMovies_No_internet_service: int = 0
    StreamingMovies_Yes: int = 0
    Contract_One_year: int = 0
    Contract_Two_year: int = 0
    PaymentMethod_Credit_card_automatic: int = 0
    PaymentMethod_Electronic_check: int = 0
    PaymentMethod_Mailed_check: int = 0

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running "}

@app.post("/predict")
def predict(data: CustomerData):
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])

    
    input_df.rename(columns={
        'PaymentMethod_Credit_card_automatic': 'PaymentMethod_Credit_card_(automatic)'
    }, inplace=True)

    
    input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
        input_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_label": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 3),
        "message": "This customer is likely to churn!" if prediction == 1 else "This customer is likely to stay."
    }