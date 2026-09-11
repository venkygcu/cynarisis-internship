# Week 1 Weekly Report - AI/ML Internship

## Summary

This week I established a reproducible Python data-analysis workspace and completed the Week 1 learning activities from NumPy fundamentals through exploratory data analysis and visualisation. Each completed task is versioned on `feat/aiml-W1-venky` with runnable source code, notebooks, and supporting artifacts.

## What I learned

- Used NumPy arrays for shapes, broadcasting, vectorised calculations, matrix multiplication, and descriptive statistics.
- Used Pandas to load CSV data, inspect schema and nulls, normalise columns, coerce types, remove duplicates, and export cleaned CSV/Parquet datasets.
- Performed EDA with `describe`, `info`, and null-count checks; documented observations, imputed missing ML scores with the median, and engineered average-score and grade features.
- Created reproducible Matplotlib and Seaborn histograms, scatter plots, correlation heatmaps, and category-count charts.
- Practised Git workflow: feature branching, descriptive commits, committed notebooks/artifacts, and remote pushes.

## What I built

| Day | Deliverable | Evidence |
| --- | --- | --- |
| W1D1 | NumPy fundamentals script and notebook | `src/w1d1.py`, `notebooks/w1d1.ipynb` |
| W1D2-W1D3 | India life-expectancy data cleaning workflow and exports | `notebooks/w1d2.ipynb`, `notebooks/w1d3.ipynb`, `data/w1d*_cleaned_*` |
| W1D4 | Student-performance EDA, 200-word narrative, and plots | `src/w1d4_eda.py`, `notebooks/w1d4.ipynb`, `artifacts/eda/` |
| W1D5 | Matplotlib/Seaborn visualization module, tests, notebook, and charts | `src/w1d5_visualization.py`, `tests/test_w1d5_visualization.py`, `notebooks/w1d5.ipynb`, `artifacts/w1d5_visualization/` |

## Validation and MLOps alignment

The W1D5 visualisation test suite passes with `3 passed`. W1D4 and W1D5 use deterministic dataset generation, reusable functions, executed notebooks, saved artifacts, and Git-tracked deliverables. These are MLOps-ready foundations for future MLflow experiment tracking and pipeline automation. CrewAI, LangGraph, and Ragas were not required for this week's deterministic data-analysis and visualisation tasks; they will be applied when a task includes agent orchestration, workflow graphs, or LLM evaluation.

## Blockers and resolutions

- The browser-based Pyodide runtime did not initially include Pandas. The local virtual environment was used for the completed work; in Pyodide, packages must be loaded before importing them.
- Seaborn was not initially installed. It was added to `requirements.txt` and pinned at version `0.13.2`.
- The local GitHub CLI/browser session was not authenticated for PR creation. All commits were pushed successfully; the PR can be opened after signing in.

## Plan for next week

1. Review the next lesson objectives and prepare the development environment.
2. Build the next data/ML task with reusable modules and tests from the start.
3. Add MLflow tracking when model training is introduced.
4. Open the pending GitHub pull request after authentication and respond promptly to review feedback.
