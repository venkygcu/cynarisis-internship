"""W3D5: compare GridSearchCV and RandomizedSearchCV for SVM and KNN."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import loguniform
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
DEFAULT_OUTPUT_DIR = Path("outputs/w3d5_hyperparameter_tuning")
CV_FOLDS = 5
RANDOM_SEARCH_ITERATIONS = 10


def load_split(random_state: int = RANDOM_STATE):
    """Load the breast-cancer data and create a stratified holdout."""
    features, target = load_breast_cancer(return_X_y=True, as_frame=True)
    return train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=random_state,
    )


def search_spaces() -> dict[str, dict[str, Any]]:
    """Return preprocessing pipelines and matched grid/random search spaces."""
    return {
        "svm": {
            "pipeline": Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", SVC(random_state=RANDOM_STATE)),
                ]
            ),
            "grid": [
                {"model__kernel": ["linear"], "model__C": [0.01, 0.1, 1, 10, 100]},
                {
                    "model__kernel": ["rbf"],
                    "model__C": [0.01, 0.1, 1, 10, 100],
                    "model__gamma": ["scale", "auto"],
                },
            ],
            "random": [
                {"model__kernel": ["linear"], "model__C": loguniform(0.01, 100)},
                {
                    "model__kernel": ["rbf"],
                    "model__C": loguniform(0.01, 100),
                    "model__gamma": loguniform(1e-4, 1),
                },
            ],
        },
        "knn": {
            "pipeline": Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", KNeighborsClassifier()),
                ]
            ),
            "grid": {
                "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
            "random": {
                "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 25, 31],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        },
    }


def evaluate(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    """Refit a selected estimator and score train and held-out data."""
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    scores = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )
    return {
        "train_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
        "test_accuracy": float(accuracy_score(y_test, prediction)),
        "test_f1": float(f1_score(y_test, prediction)),
        "test_roc_auc": float(roc_auc_score(y_test, scores)),
    }


def log_to_mlflow(
    output_dir: Path,
    results: dict[str, Any],
    artifacts: tuple[Path, ...],
) -> bool:
    """Track search settings, CV/holdout metrics, and evidence in MLflow."""
    try:
        import mlflow

        tracking_db = (output_dir / "mlflow.db").resolve().as_posix()
        mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
        mlflow.set_experiment("w3d5_hyperparameter_tuning")
        with mlflow.start_run(run_name="grid-vs-random-search"):
            mlflow.log_param("cv_folds", CV_FOLDS)
            mlflow.log_param("random_search_iterations", RANDOM_SEARCH_ITERATIONS)
            for name, result in results.items():
                mlflow.log_params(
                    {f"{name}_{key}": value for key, value in result["parameters"].items()}
                )
                mlflow.log_metrics(
                    {f"{name}_{key}": value for key, value in result["metrics"].items()}
                )
                mlflow.log_metric(
                    f"{name}_best_cv_roc_auc", result["best_cv_roc_auc"]
                )
            for artifact in artifacts:
                mlflow.log_artifact(str(artifact), artifact_path="evidence")
    except Exception as error:
        warnings.warn(
            f"MLflow tracking was skipped: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


def run_experiment(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    track_mlflow: bool = True,
) -> dict[str, Any]:
    """Compare four searches, select by CV, then report holdout performance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_split()
    spaces = search_spaces()
    results: dict[str, Any] = {}
    cv_tables: list[pd.DataFrame] = []

    for model_name, config in spaces.items():
        for method in ("grid", "random"):
            name = f"{model_name}_{method}"
            if method == "grid":
                search = GridSearchCV(
                    config["pipeline"],
                    config["grid"],
                    scoring="roc_auc",
                    cv=CV_FOLDS,
                    n_jobs=1,
                    refit=True,
                )
                candidate_count = len(ParameterGrid(config["grid"]))
            else:
                search = RandomizedSearchCV(
                    config["pipeline"],
                    config["random"],
                    n_iter=RANDOM_SEARCH_ITERATIONS,
                    scoring="roc_auc",
                    cv=CV_FOLDS,
                    n_jobs=1,
                    random_state=RANDOM_STATE,
                    refit=True,
                )
                candidate_count = RANDOM_SEARCH_ITERATIONS

            search.fit(X_train, y_train)
            results[name] = {
                "model": model_name,
                "search_method": method,
                "candidate_count": candidate_count,
                "parameters": search.best_params_,
                "best_cv_roc_auc": float(search.best_score_),
                "metrics": evaluate(
                    search.best_estimator_, X_train, X_test, y_train, y_test
                ),
            }
            cv_frame = pd.DataFrame(search.cv_results_)
            cv_frame.insert(0, "experiment", name)
            cv_tables.append(
                cv_frame[["experiment", "params", "mean_test_score", "std_test_score", "rank_test_score"]]
            )

    # Select the model/search strategy from training-fold CV, not the holdout.
    recommended = max(results, key=lambda key: results[key]["best_cv_roc_auc"])
    rows = [
        {
            "experiment": name,
            "model": result["model"],
            "search_method": result["search_method"],
            "candidate_count": result["candidate_count"],
            "best_cv_roc_auc": result["best_cv_roc_auc"],
            **result["metrics"],
            "parameters": json.dumps(result["parameters"], sort_keys=True),
        }
        for name, result in results.items()
    ]
    summary_frame = pd.DataFrame(rows)
    summary_path = output_dir / "search_comparison.csv"
    summary_frame.to_csv(summary_path, index=False)
    cv_results_path = output_dir / "cv_results.csv"
    pd.concat(cv_tables, ignore_index=True).to_csv(cv_results_path, index=False)

    figure, axis = plt.subplots(figsize=(9, 5))
    summary_frame.set_index("experiment")[["best_cv_roc_auc", "test_roc_auc"]].plot(
        kind="bar", ax=axis, color=["#3976af", "#e28e2c"]
    )
    axis.set_ylim(0.8, 1.01)
    axis.set_ylabel("ROC-AUC")
    axis.set_title("Grid Search vs Random Search: SVM and KNN")
    axis.tick_params(axis="x", rotation=0)
    axis.legend(["Best 5-fold CV", "Held-out test"])
    figure.tight_layout()
    plot_path = output_dir / "search_comparison.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)

    summary = {
        "dataset": "sklearn breast cancer",
        "random_state": RANDOM_STATE,
        "holdout_fraction": 0.2,
        "cv_folds": CV_FOLDS,
        "random_search_iterations": RANDOM_SEARCH_ITERATIONS,
        "selection_metric": "best 5-fold CV ROC-AUC",
        "recommended_experiment": recommended,
        "results": results,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if track_mlflow:
        log_to_mlflow(
            output_dir,
            results,
            (summary_path, cv_results_path, plot_path, metrics_path),
        )
    return summary


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
