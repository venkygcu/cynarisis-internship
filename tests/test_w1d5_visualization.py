"""Tests for the reproducible W1D5 visualisation workflow."""

from pathlib import Path

import pytest

from src.w1d5_visualization import build_student_dataframe, create_visualizations, validate_dataframe


def test_build_student_dataframe_is_ready_for_plotting() -> None:
    """The shared W1D4 preparation creates a complete visualisation dataset."""
    df = build_student_dataframe()

    assert df.shape == (20, 9)
    assert df[["math", "python", "ml_score", "avg_score"]].isna().sum().sum() == 0
    assert df["grade"].value_counts().to_dict() == {"C": 11, "B": 7, "F": 2, "A": 0}


def test_create_visualizations_writes_all_expected_figures(tmp_path: Path) -> None:
    """Every required Matplotlib/Seaborn chart is saved as a non-empty PNG."""
    output_paths = create_visualizations(build_student_dataframe(), tmp_path)

    assert [path.name for path in output_paths] == [
        "score_distributions.png",
        "score_relationship.png",
        "correlation_heatmap.png",
        "category_counts.png",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in output_paths)


def test_validate_dataframe_rejects_missing_chart_columns() -> None:
    """Clear validation errors prevent incomplete data from producing charts."""
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(build_student_dataframe().drop(columns="grade"))
