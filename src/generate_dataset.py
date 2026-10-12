from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
ROW_COUNT = 30000

np.random.seed(RANDOM_SEED)

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "customer_churn_raw.csv"
)


customer_ids = [
    f"CUST-{index:06d}"
    for index in range(1, ROW_COUNT + 1)
]


tenure_months = np.random.randint(
    1,
    73,
    ROW_COUNT,
)

monthly_charges = np.round(
    np.random.uniform(
        20,
        150,
        ROW_COUNT,
    ),
    2,
)

support_tickets = np.random.poisson(
    1.8,
    ROW_COUNT,
)

login_frequency = np.random.randint(
    1,
    31,
    ROW_COUNT,
)

contract_type = np.random.choice(
    [
        "Monthly",
        "One Year",
        "Two Year",
    ],
    size=ROW_COUNT,
    p=[
        0.55,
        0.25,
        0.20,
    ],
)

payment_method = np.random.choice(
    [
        "Credit Card",
        "Debit Card",
        "Bank Transfer",
        "Electronic Check",
    ],
    size=ROW_COUNT,
)

internet_service = np.random.choice(
    [
        "Fiber",
        "DSL",
        "None",
    ],
    size=ROW_COUNT,
    p=[
        0.50,
        0.35,
        0.15,
    ],
)

has_partner = np.random.choice(
    [
        "Yes",
        "No",
    ],
    size=ROW_COUNT,
)

auto_pay = np.random.choice(
    [
        "Yes",
        "No",
    ],
    size=ROW_COUNT,
)

satisfaction_score = np.random.randint(
    1,
    6,
    ROW_COUNT,
)


risk_score = (
    (contract_type == "Monthly") * 1.5
    + (support_tickets >= 3) * 1.3
    + (login_frequency <= 5) * 1.2
    + (satisfaction_score <= 2) * 1.8
    + (monthly_charges >= 100) * 0.8
    + (auto_pay == "No") * 0.6
    - (tenure_months >= 36) * 1.0
    - (contract_type == "Two Year") * 1.4
)


probability = 1 / (
    1 + np.exp(
        -(risk_score - 2.0)
    )
)


churn = np.where(
    np.random.random(
        ROW_COUNT
    ) < probability,
    "Yes",
    "No",
)


df = pd.DataFrame(
    {
        "customer_id": customer_ids,
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
        "churn": churn,
    }
)


# Intentional data quality issues

missing_rows = np.random.choice(
    df.index,
    size=250,
    replace=False,
)

df.loc[
    missing_rows,
    "payment_method",
] = np.nan


messy_contract_rows = np.random.choice(
    df.index,
    size=150,
    replace=False,
)

df.loc[
    messy_contract_rows,
    "contract_type",
] = (
    df.loc[
        messy_contract_rows,
        "contract_type",
    ]
    .str.lower()
)


duplicate_rows = df.sample(
    100,
    random_state=RANDOM_SEED,
)

df = pd.concat(
    [
        df,
        duplicate_rows,
    ],
    ignore_index=True,
)


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTPUT_PATH,
    index=False,
)


print("=" * 60)
print("CUSTOMER CHURN DATASET GENERATED")
print("=" * 60)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Output: {OUTPUT_PATH}")

print()
print("Churn distribution:")
print(df["churn"].value_counts())