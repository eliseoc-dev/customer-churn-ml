from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "customer_churn_raw.csv"
)

CLEAN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_clean.csv"
)


def clean_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)

    print("=" * 60)
    print("CUSTOMER CHURN DATA CLEANING")
    print("=" * 60)

    print(f"Initial rows: {len(df):,}")
    print(f"Duplicates: {df.duplicated().sum()}")

    df = df.drop_duplicates().copy()

    text_columns = [
        "contract_type",
        "payment_method",
        "internet_service",
        "has_partner",
        "auto_pay",
        "churn",
    ]

    for column in text_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    df["contract_type"] = (
        df["contract_type"]
        .str.title()
    )

    df["payment_method"] = (
        df["payment_method"]
        .fillna("Unknown")
    )

    numeric_columns = [
        "tenure_months",
        "monthly_charges",
        "support_tickets",
        "login_frequency",
        "satisfaction_score",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df[
        (df["tenure_months"] > 0)
        & (df["monthly_charges"] > 0)
        & (df["support_tickets"] >= 0)
        & (df["login_frequency"] >= 0)
        & (df["satisfaction_score"].between(1, 5))
    ].copy()

    df["churn_flag"] = (
        df["churn"]
        .map(
            {
                "Yes": 1,
                "No": 0,
            }
        )
    )

    df["high_support_usage"] = (
        df["support_tickets"] >= 3
    ).astype(int)

    df["low_engagement"] = (
        df["login_frequency"] <= 5
    ).astype(int)

    df["high_monthly_charge"] = (
        df["monthly_charges"] >= 100
    ).astype(int)

    df["long_tenure"] = (
        df["tenure_months"] >= 36
    ).astype(int)

    CLEAN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        CLEAN_PATH,
        index=False,
    )

    print(f"Final rows: {len(df):,}")
    print(f"Saved to: {CLEAN_PATH}")

    return df


if __name__ == "__main__":
    clean_data()