"""Leakage-safe, reproducible preprocessing pipeline for Week 2.

Run from the repository root:
    .venv\\Scripts\\python.exe src\\w2_preprocessing_pipeline.py

The pipeline validates the raw schema, splits before fitting any transformer,
and persists concise data-quality and model-validation evidence.  MLflow is
used when available; the core preprocessing workflow remains runnable without
network access or an LLM provider.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "seaborn" / "titanic.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "w2_preprocessing"
TARGET_COLUMN = "survived"
NUMERIC_COLUMNS = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_COLUMNS = ["sex", "embarked", "class", "who", "alone"]
REQUIRED_COLUMNS = set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN])


@dataclass(frozen=True)
class RunSummary:
    """Serializable evidence for one reproducible pipeline run."""

    train_rows: int
    test_rows: int
    raw_rows: int
    transformed_features: int
    train_positive_rate: float
    test_positive_rate: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def load_dataset(dataset_path: Path = DEFAULT_DATASET) -> pd.DataFrame:
    """Load the supplied Titanic CSV and fail clearly for an absent source."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return pd.read_csv(dataset_path)


def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate required columns, binary target values, and non-empty input."""
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset must contain at least one row.")
    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"{TARGET_COLUMN} must not contain missing values.")
    target_values = set(df[TARGET_COLUMN].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"{TARGET_COLUMN} must be binary 0/1; found {sorted(target_values)}")


def build_preprocessor() -> ColumnTransformer:
    """Create train-fitted imputing, scaling, and encoding transformations."""
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


def build_pipeline() -> Pipeline:
    """Build the complete preprocessing-plus-classification pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1_000, random_state=42)),
        ]
    )


def quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize source quality before transformations modify the data."""
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {column: int(value) for column, value in df.isna().sum().items()},
        "target_distribution": {
            str(label): int(count)
            for label, count in df[TARGET_COLUMN].value_counts(dropna=False).sort_index().items()
        },
    }


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calculate holdout metrics from a fitted pipeline."""
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def _log_to_mlflow(summary: RunSummary, quality: dict[str, Any], output_dir: Path) -> None:
    """Log local MLflow evidence without making MLflow a runtime requirement."""
    try:
        import mlflow
    except ImportError:
        return

    # MLflow 3 deprecates its file store.  A local SQLite database keeps the
    # run self-contained and avoids requiring a remote tracking server.
    tracking_uri = f"sqlite:///{(output_dir / 'mlflow.db').resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("week2-preprocessing")
    with mlflow.start_run(run_name="titanic-leakage-safe-preprocessing"):
        mlflow.log_params({"test_size": 0.2, "random_state": 42, "model": "logistic_regression"})
        mlflow.log_metrics({key: value for key, value in asdict(summary).items() if isinstance(value, float)})
        quality_path = output_dir / "data_quality_report.json"
        mlflow.log_artifact(str(quality_path), artifact_path="evidence")


def run_pipeline(dataset_path: Path = DEFAULT_DATASET, output_dir: Path = DEFAULT_OUTPUT_DIR) -> RunSummary:
    """Execute the end-to-end workflow and persist portable output evidence."""
    raw_df = load_dataset(dataset_path)
    validate_raw_data(raw_df)
    output_dir.mkdir(parents=True, exist_ok=True)
    quality = quality_report(raw_df)

    X = raw_df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()
    y = raw_df[TARGET_COLUMN].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()

    summary = RunSummary(
        raw_rows=len(raw_df),
        train_rows=len(X_train),
        test_rows=len(X_test),
        transformed_features=len(feature_names),
        train_positive_rate=float(y_train.mean()),
        test_positive_rate=float(y_test.mean()),
        **metrics,
    )
    (output_dir / "data_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    (output_dir / "run_summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    pd.DataFrame({"feature": feature_names}).to_csv(output_dir / "transformed_features.csv", index=False)
    joblib.dump(pipeline, output_dir / "preprocessing_model.joblib")
    _log_to_mlflow(summary, quality, output_dir)
    return summary


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(asdict(result), indent=2))
