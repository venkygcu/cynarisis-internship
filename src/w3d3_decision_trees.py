"""W3D3: reproducible Decision Tree and Random Forest classification workflow.

The workflow makes overfitting visible by comparing an unrestricted tree with a
cross-validated, depth-limited tree.  MLflow records parameters, metrics, and
the generated artifacts so the experiment can be reproduced or promoted later.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

RANDOM_STATE = 42
DEFAULT_OUTPUT_DIR = Path("outputs/w3d3_decision_trees")


def log_to_mlflow(
    output_dir: Path,
    tree_parameters: dict[str, Any],
    forest_parameters: dict[str, Any],
    baseline_metrics: dict[str, float],
    tuned_tree_metrics: dict[str, float],
    forest_metrics: dict[str, float],
    artifacts: tuple[Path, ...],
) -> bool:
    """Record evidence in MLflow without making training depend on its import.

    A notebook kernel can retain a partially imported third-party module after
    an interrupted run.  Delaying the MLflow import lets the core exercise run
    and provides an actionable warning instead of masking model results.
    """
    try:
        import mlflow

        mlflow.set_tracking_uri((output_dir / "mlruns").resolve().as_uri())
        mlflow.set_experiment("w3d3_decision_trees")
        with mlflow.start_run(run_name="tree-and-forest-tuning"):
            mlflow.log_params({f"tree_{key}": value for key, value in tree_parameters.items()})
            mlflow.log_params({f"forest_{key}": value for key, value in forest_parameters.items()})
            mlflow.log_metrics({f"baseline_{key}": value for key, value in baseline_metrics.items()})
            mlflow.log_metrics({f"tuned_tree_{key}": value for key, value in tuned_tree_metrics.items()})
            mlflow.log_metrics({f"forest_{key}": value for key, value in forest_metrics.items()})
            for artifact in artifacts:
                mlflow.log_artifact(str(artifact), artifact_path="evidence")
    except (AttributeError, ImportError) as error:
        warnings.warn(
            f"MLflow tracking was skipped because MLflow could not initialise: {error}. "
            "Restart the notebook kernel and run the cells again to enable tracking.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


def gini_impurity(class_counts: list[int] | np.ndarray) -> float:
    """Return Gini impurity, safely handling an empty node."""
    counts = np.asarray(class_counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    probabilities = counts / total
    return float(1 - np.square(probabilities).sum())


def information_gain(
    parent_counts: list[int], left_counts: list[int], right_counts: list[int]
) -> float:
    """Compute the Gini reduction delivered by a candidate binary split."""
    parent_total = sum(parent_counts)
    if parent_total == 0:
        return 0.0
    return gini_impurity(parent_counts) - (
        sum(left_counts) / parent_total * gini_impurity(left_counts)
        + sum(right_counts) / parent_total * gini_impurity(right_counts)
    )


def load_split(random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the breast-cancer data and create a stratified 80/20 holdout."""
    # Request the explicit tuple form. This avoids ambiguous Bunch typing in
    # editors while retaining labelled DataFrame/Series inputs for the plots.
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    split = train_test_split(
        pd.DataFrame(X),
        pd.Series(y),
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = split
    return pd.DataFrame(X_train), pd.DataFrame(X_test), pd.Series(y_train), pd.Series(y_test)


def evaluate(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict[str, float]:
    """Fit a classifier and return train/test metrics used to spot overfitting."""
    model.fit(X_train, y_train)
    test_probability = model.predict_proba(X_test)[:, 1]
    return {
        "train_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
        "test_accuracy": float(accuracy_score(y_test, model.predict(X_test))),
        "test_f1": float(f1_score(y_test, model.predict(X_test))),
        "test_roc_auc": float(roc_auc_score(y_test, test_probability)),
    }


def save_tree_plot(model: DecisionTreeClassifier, feature_names: list[str], path: Path) -> None:
    """Render the selected tree, keeping its depth readable for the report."""
    figure, axis = plt.subplots(figsize=(18, 9))
    plot_tree(model, feature_names=feature_names, class_names=["malignant", "benign"], filled=True, rounded=True, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_feature_importance(model: RandomForestClassifier, feature_names: list[str], path: Path) -> None:
    """Save the ten most influential forest features."""
    importance = pd.Series(model.feature_importances_, index=feature_names).nlargest(10).sort_values()
    figure, axis = plt.subplots(figsize=(8, 5))
    importance.plot.barh(ax=axis, color="#3976af")
    axis.set_title("Random Forest: top 10 feature importances")
    axis.set_xlabel("Mean impurity decrease")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_experiment(output_dir: Path | str = DEFAULT_OUTPUT_DIR, track_mlflow: bool = True) -> dict[str, Any]:
    """Train, tune, track, and persist the complete W3D3 learning experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_split()

    baseline = DecisionTreeClassifier(random_state=RANDOM_STATE)
    baseline_metrics = evaluate(baseline, X_train, X_test, y_train, y_test)

    tree_search = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid={"criterion": ["gini", "entropy"], "max_depth": [2, 3, 4, 5, 6], "min_samples_leaf": [1, 3, 5, 10]},
        scoring="roc_auc",
        cv=5,
        n_jobs=1,
    )
    tree_search.fit(X_train, y_train)
    tuned_tree = tree_search.best_estimator_
    tuned_tree_metrics = evaluate(tuned_tree, X_train, X_test, y_train, y_test)

    forest_search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        param_grid={"n_estimators": [100, 200], "max_depth": [None, 4, 6], "min_samples_leaf": [1, 3]},
        scoring="roc_auc",
        cv=5,
        n_jobs=1,
    )
    forest_search.fit(X_train, y_train)
    forest = forest_search.best_estimator_
    forest_metrics = evaluate(forest, X_train, X_test, y_train, y_test)

    tree_path, importance_path = output_dir / "tuned_tree.png", output_dir / "forest_feature_importance.png"
    save_tree_plot(tuned_tree, list(X_train.columns), tree_path)
    save_feature_importance(forest, list(X_train.columns), importance_path)
    cv_results = pd.DataFrame(tree_search.cv_results_)[["params", "mean_test_score", "rank_test_score"]].sort_values("rank_test_score")
    cv_results.to_csv(output_dir / "tree_cv_results.csv", index=False)

    summary = {
        "baseline_tree": baseline_metrics,
        "tuned_tree": {**tuned_tree_metrics, "best_cv_roc_auc": tree_search.best_score_, "parameters": tree_search.best_params_},
        "random_forest": {**forest_metrics, "best_cv_roc_auc": forest_search.best_score_, "parameters": forest_search.best_params_},
        "overfit_gap_reduction": baseline_metrics["train_accuracy"] - baseline_metrics["test_accuracy"] - (tuned_tree_metrics["train_accuracy"] - tuned_tree_metrics["test_accuracy"]),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if track_mlflow:
        log_to_mlflow(
            output_dir,
            tree_search.best_params_,
            forest_search.best_params_,
            baseline_metrics,
            tuned_tree_metrics,
            forest_metrics,
            (tree_path, importance_path, output_dir / "tree_cv_results.csv", output_dir / "metrics.json"),
        )
    return summary


if __name__ == "__main__":
    results = run_experiment()
    print(json.dumps(results, indent=2))
