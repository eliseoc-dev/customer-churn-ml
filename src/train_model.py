from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_clean.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


df = pd.read_csv(DATA_PATH)


TARGET = "churn_flag"

DROP_COLUMNS = [
    "customer_id",
    "churn",
    TARGET,
]


X = df.drop(
    columns=DROP_COLUMNS
)

y = df[TARGET]


categorical_features = [
    "contract_type",
    "payment_method",
    "internet_service",
    "has_partner",
    "auto_pay",
]


numerical_features = [
    column
    for column in X.columns
    if column not in categorical_features
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numerical_features,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features,
        ),
    ]
)


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
)


models = {
    "logistic_regression":
        LogisticRegression(
            max_iter=1000
        ),

    "random_forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
}


results = {}


for model_name, model in models.items():
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                model,
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
    )

    recall = recall_score(
        y_test,
        predictions,
    )

    f1 = f1_score(
        y_test,
        predictions,
    )

    results[model_name] = {
        "pipeline": pipeline,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    print()
    print("=" * 60)
    print(model_name.upper())
    print("=" * 60)

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print()
    print(
        classification_report(
            y_test,
            predictions,
        )
    )


best_model_name = max(
    results,
    key=lambda name:
        results[name]["f1"],
)

best_pipeline = (
    results[
        best_model_name
    ]["pipeline"]
)


MODEL_PATH = (
    MODEL_DIR
    / "churn_model.joblib"
)

joblib.dump(
    best_pipeline,
    MODEL_PATH,
)


print()
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    f"Selected: "
    f"{best_model_name}"
)

print(
    f"Saved to: "
    f"{MODEL_PATH}"
)