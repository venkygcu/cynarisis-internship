"""W1D4 EDA on a reproducible student-performance dataset.

Run from the repository root:
    .venv\\Scripts\\python.exe src\\w1d4_eda.py

For Pyodide, load packages first:
    import pyodide
    await pyodide.loadPackage(["pandas", "numpy", "matplotlib"])
"""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "eda"


def create_student_dataset() -> pd.DataFrame:
    """Create the supplied 20-student dataset with reproducible missing values."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "student_id": range(1, 21),
            "name": [f"Student_{number}" for number in range(1, 21)],
            "age": np.random.randint(20, 26, 20),
            "math": np.random.randint(50, 100, 20),
            "python": np.random.randint(45, 100, 20),
            "ml_score": np.random.randint(40, 100, 20),
            "attended": np.random.choice([True, False], 20, p=[0.8, 0.2]),
        }
    )
    df.loc[[3, 7, 14], "ml_score"] = np.nan
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Run, print, and save the three required EDA inspection commands."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "inspection.txt").open("w", encoding="utf-8") as report:
        with redirect_stdout(report):
            print("=== Dataset Overview ===")
            print(df.head().to_string(index=False))
            print(f"\nShape: {df.shape}")
            print("\ndf.describe()")
            print(df.describe(include="all").to_string())
            print("\ndf.info()")
            info_output = StringIO()
            df.info(buf=info_output)
            print("\n".join(line.rstrip() for line in info_output.getvalue().splitlines()))
            print("\ndf.isnull().sum()")
            print(df.isnull().sum().to_string())

    print("=== Dataset Overview ===")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print("\ndf.describe()")
    print(df.describe(include="all"))
    print("\ndf.info()")
    df.info()
    print("\ndf.isnull().sum()")
    print(df.isnull().sum())


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Impute the ML median and create the requested average-score grade."""
    cleaned = df.copy()
    cleaned["ml_score"] = cleaned["ml_score"].fillna(cleaned["ml_score"].median())
    cleaned["avg_score"] = cleaned[["math", "python", "ml_score"]].mean(axis=1).round(1)
    cleaned["grade"] = pd.cut(
        cleaned["avg_score"],
        bins=[0, 59, 74, 89, 100],
        labels=["F", "C", "B", "A"],
        include_lowest=True,
    )
    return cleaned


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """Plot a distribution for every numeric column, including engineered scores."""
    numeric = df.select_dtypes(include="number")
    axes = numeric.hist(bins=8, figsize=(14, 10), edgecolor="white", color="#2c7fb8")
    for axis in axes.flat:
        axis.set_ylabel("Students")
    plt.suptitle("Student dataset: numeric-column distributions", y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "numeric_distributions.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot Pearson correlations for numeric variables."""
    correlation = df.select_dtypes(include="number").corr()
    fig, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation)), correlation.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(correlation)), correlation.index)
    fig.colorbar(image, ax=axis, label="Pearson correlation")
    axis.set_title("Student dataset: correlation heatmap")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_top_categories(df: pd.DataFrame) -> None:
    """Plot the top ten counts for the meaningful categorical features."""
    columns = ["grade", "attended"]
    fig, axes = plt.subplots(1, len(columns), figsize=(10, 4))
    for axis, column in zip(axes, columns):
        df[column].value_counts().head(10).sort_values().plot.barh(ax=axis, color="#41ab5d")
        axis.set_title(f"Top categories: {column}")
        axis.set_xlabel("Students")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_10_category_counts.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_feature_summary(df: pd.DataFrame) -> None:
    """Print the assignment's requested grade, ranking, and attendance summaries."""
    print("\n=== Grade Distribution ===")
    print(df["grade"].value_counts().sort_index())
    print("\n=== Top 5 Students ===")
    print(df.nlargest(5, "avg_score")[["name", "avg_score", "grade"]])
    print("\n=== Attendees vs Non-Attendees ===")
    print(df.groupby("attended")["avg_score"].agg(["mean", "count"]).round(2))


def main() -> None:
    raw_df = create_student_dataset()
    inspect_data(raw_df)
    df = engineer_features(raw_df)
    plot_numeric_distributions(df)
    plot_correlation_heatmap(df)
    plot_top_categories(df)
    print_feature_summary(df)
    print(f"\nSaved EDA artifacts to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
