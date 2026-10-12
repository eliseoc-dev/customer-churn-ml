from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_clean.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "churn_model.joblib"
)

REPORTS_PATH = (
    PROJECT_ROOT
    / "reports"
)

REPORTS_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


df = pd.read_csv(DATA_PATH)

TARGET = "churn_flag"

X = df.drop(
    columns=[
        "customer_id",
        "churn",
        TARGET,
    ]
)

y = df[TARGET]


_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


model = joblib.load(MODEL_PATH)

predictions = model.predict(X_test)

probabilities = model.predict_proba(
    X_test
)[:, 1]


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

roc_auc = roc_auc_score(
    y_test,
    probabilities,
)


print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print()
print(
    classification_report(
        y_test,
        predictions,
    )
)


ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
)

plt.title("Churn Confusion Matrix")
plt.tight_layout()

plt.savefig(
    REPORTS_PATH
    / "confusion_matrix.png"
)

plt.close()


RocCurveDisplay.from_predictions(
    y_test,
    probabilities,
)

plt.title("Churn ROC Curve")
plt.tight_layout()

plt.savefig(
    REPORTS_PATH
    / "roc_curve.png"
)

plt.close()


print()
print("Saved:")
print("confusion_matrix.png")
print("roc_curve.png")