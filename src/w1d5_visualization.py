"""W1D5 Matplotlib and Seaborn visualisations for student performance.

Run from the repository root:
    .venv\\Scripts\\python.exe src\\w1d5_visualization.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from src.w1d4_eda import create_student_dataset, engineer_features
except ModuleNotFoundError:  # Supports `python src/w1d5_visualization.py`.
    from w1d4_eda import create_student_dataset, engineer_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "w1d5_visualization"
SCORE_COLUMNS = ["math", "python", "ml_score", "avg_score"]


def build_student_dataframe() -> pd.DataFrame:
    """Reuse W1D4's deterministic data preparation for comparable charts."""
    return engineer_features(create_student_dataset())


def validate_dataframe(df: pd.DataFrame) -> None:
    """Fail early when a chart would be built from an incomplete table."""
    required_columns = {"math", "python", "ml_score", "avg_score", "grade", "attended"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if df[SCORE_COLUMNS].isna().any().any():
        raise ValueError("Score columns must be imputed before visualisation.")


def plot_score_distributions(df: pd.DataFrame, output_dir: Path) -> Path:
    """Create one Seaborn histogram for every student score column."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    palette = ["#2563eb", "#7c3aed", "#db2777", "#059669"]
    for axis, column, color in zip(axes.flat, SCORE_COLUMNS, palette):
        sns.histplot(df[column], bins=7, kde=True, color=color, ax=axis)
        axis.set_title(f"{column.replace('_', ' ').title()} distribution")
        axis.set_xlabel("Score")
        axis.set_ylabel("Students")
    figure.suptitle("Student score distributions", fontsize=15, fontweight="bold")
    figure.tight_layout()
    path = output_dir / "score_distributions.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_score_relationship(df: pd.DataFrame, output_dir: Path) -> Path:
    """Show the mathematics/Python relationship, coloured by grade."""
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="math",
        y="python",
        hue="grade",
        style="attended",
        s=90,
        palette="viridis",
        ax=axis,
    )
    axis.set_title("Mathematics versus Python scores")
    axis.set_xlabel("Mathematics score")
    axis.set_ylabel("Python score")
    figure.tight_layout()
    path = output_dir / "score_relationship.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot annotated score correlations with a divergent Seaborn colour scale."""
    figure, axis = plt.subplots(figsize=(8, 6))
    correlation = df[SCORE_COLUMNS].corr()
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axis)
    axis.set_title("Score correlation heatmap")
    figure.tight_layout()
    path = output_dir / "correlation_heatmap.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_category_counts(df: pd.DataFrame, output_dir: Path) -> Path:
    """Compare grade and attendance frequencies in two categorical count plots."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.countplot(data=df, x="grade", order=["F", "C", "B", "A"], color="#7c3aed", ax=axes[0])
    axes[0].set_title("Grade counts")
    axes[0].set_xlabel("Grade")
    axes[0].set_ylabel("Students")
    sns.countplot(data=df, x="attended", color="#059669", ax=axes[1])
    axes[1].set_title("Attendance counts")
    axes[1].set_xlabel("Attended")
    axes[1].set_ylabel("Students")
    figure.tight_layout()
    path = output_dir / "category_counts.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def create_visualizations(df: pd.DataFrame, output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Create and return the four W1D5 chart artifacts."""
    validate_dataframe(df)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    return [
        plot_score_distributions(df, output_dir),
        plot_score_relationship(df, output_dir),
        plot_correlation_heatmap(df, output_dir),
        plot_category_counts(df, output_dir),
    ]


def main() -> None:
    """Build the data and create W1D5 artifacts from the command line."""
    chart_paths = create_visualizations(build_student_dataframe())
    print("Created visualisations:")
    for path in chart_paths:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
