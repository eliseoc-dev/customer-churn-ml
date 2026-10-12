import pandas as pd


DATA_PATH = (
    "data/processed/"
    "customer_churn_clean.csv"
)


def load_data():
    return pd.read_csv(DATA_PATH)


def test_no_duplicates():
    df = load_data()

    assert df.duplicated().sum() == 0


def test_churn_flag_is_binary():
    df = load_data()

    assert set(
        df["churn_flag"].unique()
    ).issubset({0, 1})


def test_positive_monthly_charges():
    df = load_data()

    assert (
        df["monthly_charges"] > 0
    ).all()


def test_valid_satisfaction_score():
    df = load_data()

    assert (
        df["satisfaction_score"]
        .between(1, 5)
    ).all()


def test_feature_engineering_columns_exist():
    df = load_data()

    expected = {
        "high_support_usage",
        "low_engagement",
        "high_monthly_charge",
        "long_tenure",
    }

    assert expected.issubset(
        df.columns
    )