"""Reproducible exploratory data analysis for the India life-expectancy data.

Run from the repository root:
    .venv\\Scripts\\python.exe src\\w1d4_eda.py

The script prints the required Pandas inspection calls and writes the figures and
text summary to ``artifacts/eda``.  It deliberately uses only the approved local
Python ML/data stack; MLOps tools belong after the data-quality issues identified
here have been resolved, not inside exploratory analysis itself.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("data/india_life_expectancy.csv")
OUTPUT_DIR = Path("artifacts/eda")


def save_required_inspections(df: pd.DataFrame) -> None:
    """Run and persist describe, info, and missing-value inspection output."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "inspection.txt"
    with output_path.open("w", encoding="utf-8") as report, redirect_stdout(report):
        print("df.describe()")
        print(df.describe(include="all").to_string())
        print("\n\ndf.info()")
        df.info()
        print("\n\ndf.isnull().sum()")
        print(df.isnull().sum().to_string())

    # Also show the three required calls directly in a terminal/notebook run.
    print("df.describe()")
    print(df.describe(include="all"))
    print("\ndf.info()")
    df.info()
    print("\ndf.isnull().sum()")
    print(df.isnull().sum())


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """Save one histogram per numeric feature in a compact grid."""
    numeric = df.select_dtypes(include="number")
    axes = numeric.hist(bins=8, figsize=(16, 18), edgecolor="white", color="#2c7fb8")
    for axis in axes.flat:
        axis.set_ylabel("Records")
    plt.suptitle("India life expectancy: numeric feature distributions", y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "numeric_distributions.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Save a labeled Pearson-correlation heatmap without an extra dependency."""
    correlation = df.select_dtypes(include="number").corr()
    fig, axis = plt.subplots(figsize=(15, 12))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
    axis.set_yticks(range(len(correlation.index)), correlation.index)
    fig.colorbar(image, ax=axis, label="Pearson correlation")
    axis.set_title("Correlation heatmap: India life-expectancy features")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_top_categories(df: pd.DataFrame) -> None:
    """Save top-ten count charts for every categorical field."""
    categories = df.select_dtypes(exclude="number")
    fig, axes = plt.subplots(1, len(categories.columns), figsize=(7 * len(categories.columns), 5))
    if len(categories.columns) == 1:
        axes = [axes]
    for axis, column in zip(axes, categories.columns):
        counts = categories[column].value_counts().head(10).sort_values()
        counts.plot.barh(ax=axis, color="#41ab5d")
        axis.set_title(f"Top categories: {column}")
        axis.set_xlabel("Count")
        axis.set_ylabel(column)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_10_category_counts.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    save_required_inspections(df)
    plot_numeric_distributions(df)
    plot_correlation_heatmap(df)
    plot_top_categories(df)
    print(f"\nSaved EDA artifacts to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
