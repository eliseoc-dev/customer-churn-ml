from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "churn_model.joblib"
)

REPORTS_PATH = (
    PROJECT_ROOT
    / "reports"
)


pipeline = joblib.load(MODEL_PATH)

preprocessor = pipeline.named_steps[
    "preprocessor"
]

classifier = pipeline.named_steps[
    "classifier"
]


feature_names = (
    preprocessor.get_feature_names_out()
)


if hasattr(
    classifier,
    "feature_importances_",
):
    importance_values = (
        classifier.feature_importances_
    )

elif hasattr(
    classifier,
    "coef_",
):
    importance_values = (
        abs(
            classifier.coef_[0]
        )
    )

else:
    raise ValueError(
        "Selected model does not expose feature importance."
    )


importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": importance_values,
    }
)


importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False,
    )
)


print("=" * 60)
print("TOP MODEL FEATURES")
print("=" * 60)

print(
    importance_df.head(15)
)


top_features = (
    importance_df
    .head(15)
    .sort_values(
        "importance"
    )
)


plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_features["feature"],
    top_features["importance"],
)

plt.title(
    "Top Churn Prediction Features"
)

plt.xlabel(
    "Importance"
)

plt.tight_layout()

plt.savefig(
    REPORTS_PATH
    / "feature_importance.png"
)

plt.close()


importance_df.to_csv(
    REPORTS_PATH
    / "feature_importance.csv",
    index=False,
)


print()
print(
    "Saved feature_importance.png"
)

print(
    "Saved feature_importance.csv"
)