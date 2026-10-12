import pandas as pd


DATA_PATH = (
    "data/raw/"
    "customer_churn_raw.csv"
)


df = pd.read_csv(DATA_PATH)


print("=" * 60)
print("CUSTOMER CHURN DATA OVERVIEW")
print("=" * 60)

print()
print("Shape:")
print(df.shape)

print()
print("Data types:")
print(df.dtypes)

print()
print("Missing values:")
print(
    df.isna()
    .sum()
    .sort_values(
        ascending=False
    )
)

print()
print("Duplicates:")
print(df.duplicated().sum())

print()
print("Churn distribution:")
print(
    df["churn"]
    .value_counts(
        normalize=True
    )
    .mul(100)
    .round(2)
)