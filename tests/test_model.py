from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "churn_model.joblib"
)


def test_model_exists():
    assert MODEL_PATH.exists()


def test_model_can_predict():
    model = joblib.load(
        MODEL_PATH
    )

    customer = pd.DataFrame(
        [
            {
                "tenure_months": 12,
                "monthly_charges": 75.0,
                "support_tickets": 1,
                "login_frequency": 15,
                "contract_type": "Monthly",
                "payment_method": "Credit Card",
                "internet_service": "Fiber",
                "has_partner": "Yes",
                "auto_pay": "Yes",
                "satisfaction_score": 4,
                "high_support_usage": 0,
                "low_engagement": 0,
                "high_monthly_charge": 0,
                "long_tenure": 0,
            }
        ]
    )

    prediction = model.predict(
        customer
    )

    assert prediction[0] in [
        0,
        1,
    ]