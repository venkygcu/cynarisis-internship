"""Tests for the Week 2 leakage-safe preprocessing workflow."""

import json

import pandas as pd
import pytest

from src.w2_preprocessing_pipeline import (
    DEFAULT_DATASET,
    REQUIRED_COLUMNS,
    build_pipeline,
    load_dataset,
    run_pipeline,
    validate_raw_data,
)


def test_pipeline_trains_and_transforms_without_missing_values() -> None:
    """The fitted transformer creates a numeric, complete feature matrix."""
    df = load_dataset(DEFAULT_DATASET)
    X = df.drop(columns="survived")
    y = df["survived"]
    pipeline = build_pipeline().fit(X, y)
    transformed = pipeline.named_steps["preprocessor"].transform(X)

    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] > len(REQUIRED_COLUMNS)
    assert not pd.isna(transformed).any()


def test_run_pipeline_writes_reproducible_evidence(tmp_path) -> None:
    """A full run produces model and concise validation artifacts."""
    summary = run_pipeline(output_dir=tmp_path)

    assert summary.train_rows + summary.test_rows == summary.raw_rows
    assert 0.0 <= summary.roc_auc <= 1.0
    assert abs(summary.train_positive_rate - summary.test_positive_rate) < 0.02
    assert (tmp_path / "preprocessing_model.joblib").exists()
    evidence = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert evidence["transformed_features"] == summary.transformed_features


def test_validate_raw_data_rejects_missing_schema_column() -> None:
    """Schema checks stop accidental training on incomplete inputs."""
    incomplete = load_dataset(DEFAULT_DATASET).drop(columns="fare")

    with pytest.raises(ValueError, match="fare"):
        validate_raw_data(incomplete)


def test_validate_raw_data_rejects_missing_target() -> None:
    """A null target cannot be silently coerced before a stratified split."""
    invalid_target = load_dataset(DEFAULT_DATASET).copy()
    invalid_target.loc[0, "survived"] = None

    with pytest.raises(ValueError, match="must not contain missing"):
        validate_raw_data(invalid_target)
