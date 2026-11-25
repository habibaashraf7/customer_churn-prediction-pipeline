"""FastAPI endpoint for real-time churn scoring.

Run:   uvicorn api:app --reload
Docs:  http://127.0.0.1:8000/docs

POST /predict with one customer's raw columns; returns the churn probability, risk band,
whether to contact them, and the top reasons.
"""
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.service import ChurnService

app = FastAPI(title="Churn scoring API", version="1.0",
              description="Scores one subscription customer and explains the score.")
svc = ChurnService()
YesNo = Literal["Yes", "No"]


class Customer(BaseModel):
    gender: Literal["Female", "Male"] = "Female"
    SeniorCitizen: int = Field(0, ge=0, le=1)
    Partner: YesNo = "No"
    Dependents: YesNo = "No"
    tenure: int = Field(3, ge=0, le=100, description="months as a customer")
    PhoneService: YesNo = "Yes"
    MultipleLines: Literal["Yes", "No", "No phone service"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = "No"
    OnlineBackup: Literal["Yes", "No", "No internet service"] = "No"
    DeviceProtection: Literal["Yes", "No", "No internet service"] = "No"
    TechSupport: Literal["Yes", "No", "No internet service"] = "No"
    StreamingTV: Literal["Yes", "No", "No internet service"] = "No"
    StreamingMovies: Literal["Yes", "No", "No internet service"] = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: YesNo = "Yes"
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)",
                           "Credit card (automatic)"] = "Electronic check"
    MonthlyCharges: float = Field(85.0, gt=0)
    TotalCharges: float = Field(255.0, ge=0)


@app.get("/health")
def health():
    return {"status": "ok", "threshold": svc.threshold}


@app.post("/predict")
def predict(customer: Customer):
    return svc.score_customer(customer.model_dump())
