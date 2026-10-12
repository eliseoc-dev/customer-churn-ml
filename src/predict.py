from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "churn_model.joblib"
)


model = joblib.load(
    MODEL_PATH
)


customer = pd.DataFrame(
    [
        {
            "tenure_months": 5,
            "monthly_charges": 125.00,
            "support_tickets": 4,
            "login_frequency": 3,
            "contract_type": "Monthly",
            "payment_method": "Electronic Check",
            "internet_service": "Fiber",
            "has_partner": "No",
            "auto_pay": "No",
            "satisfaction_score": 2,
            "high_support_usage": 1,
            "low_engagement": 1,
            "high_monthly_charge": 1,
            "long_tenure": 0,
        }
    ]
)


prediction = model.predict(
    customer
)[0]


probability = model.predict_proba(
    customer
)[0][1]


print("=" * 60)
print("CUSTOMER CHURN PREDICTION")
print("=" * 60)

print(
    "Prediction:",
    "CHURN"
    if prediction == 1
    else "NO CHURN",
)

print(
    f"Churn probability: "
    f"{probability:.2%}"
)