from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "churn_model.joblib"
)


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🤖",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(
        MODEL_PATH
    )


model = load_model()


st.title(
    "🤖 Customer Churn Predictor"
)

st.caption(
    "Machine learning application for estimating customer churn risk."
)


left, right = st.columns(2)


with left:
    tenure_months = st.number_input(
        "Tenure (months)",
        min_value=1,
        max_value=72,
        value=12,
    )

    monthly_charges = st.number_input(
        "Monthly charges",
        min_value=1.0,
        value=75.0,
        step=1.0,
    )

    support_tickets = st.number_input(
        "Support tickets",
        min_value=0,
        value=1,
    )

    login_frequency = st.number_input(
        "Monthly login frequency",
        min_value=0,
        value=15,
    )

    satisfaction_score = st.slider(
        "Satisfaction score",
        min_value=1,
        max_value=5,
        value=3,
    )


with right:
    contract_type = st.selectbox(
        "Contract type",
        [
            "Monthly",
            "One Year",
            "Two Year",
        ],
    )

    payment_method = st.selectbox(
        "Payment method",
        [
            "Credit Card",
            "Debit Card",
            "Bank Transfer",
            "Electronic Check",
            "Unknown",
        ],
    )

    internet_service = st.selectbox(
        "Internet service",
        [
            "Fiber",
            "DSL",
            "None",
        ],
    )

    has_partner = st.selectbox(
        "Has partner",
        [
            "Yes",
            "No",
        ],
    )

    auto_pay = st.selectbox(
        "Auto pay",
        [
            "Yes",
            "No",
        ],
    )


if st.button(
    "Predict churn risk",
    type="primary",
):
    customer = pd.DataFrame(
        [
            {
                "tenure_months": tenure_months,
                "monthly_charges": monthly_charges,
                "support_tickets": support_tickets,
                "login_frequency": login_frequency,
                "contract_type": contract_type,
                "payment_method": payment_method,
                "internet_service": internet_service,
                "has_partner": has_partner,
                "auto_pay": auto_pay,
                "satisfaction_score": satisfaction_score,

                "high_support_usage":
                    int(
                        support_tickets >= 3
                    ),

                "low_engagement":
                    int(
                        login_frequency <= 5
                    ),

                "high_monthly_charge":
                    int(
                        monthly_charges >= 100
                    ),

                "long_tenure":
                    int(
                        tenure_months >= 36
                    ),
            }
        ]
    )


    prediction = model.predict(
        customer
    )[0]


    probability = (
        model.predict_proba(
            customer
        )[0][1]
    )


    st.divider()


    st.metric(
        "Churn Probability",
        f"{probability:.1%}",
    )


    if probability >= 0.70:
        st.error(
            "HIGH RISK customer"
        )

    elif probability >= 0.40:
        st.warning(
            "MEDIUM RISK customer"
        )

    else:
        st.success(
            "LOW RISK customer"
        )


    st.write(
        "Prediction:",
        "CHURN"
        if prediction == 1
        else "NO CHURN",
    )