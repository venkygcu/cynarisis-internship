"""W3D4: compare tuned, scaled SVM and KNN classifiers with MLflow evidence."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
DEFAULT_OUTPUT_DIR = Path("outputs/w3d4_svm_knn")


def load_split(random_state: int = RANDOM_STATE):
    """Return a reproducible, stratified holdout from sklearn's cancer data."""
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)


def evaluate(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict[str, float]:
    """Fit a model and report train/test metrics using positive-class scores."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    return {
        "train_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_f1": float(f1_score(y_test, predictions)),
        "test_roc_auc": float(roc_auc_score(y_test, scores)),
    }


def model_searches() -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    """Define leakage-safe scaler/model pipelines and compact tuning grids."""
    return {
        "svm": (
            Pipeline([("scale", StandardScaler()), ("model", SVC(random_state=RANDOM_STATE))]),
            {"model__kernel": ["linear", "rbf"], "model__C": [0.1, 1, 10], "model__gamma": ["scale", "auto"]},
        ),
        "knn": (
            Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier())]),
            {"model__n_neighbors": [3, 5, 7, 9], "model__weights": ["uniform", "distance"], "model__p": [1, 2]},
        ),
    }


def log_to_mlflow(output_dir: Path, results: dict[str, Any], artifacts: tuple[Path, ...]) -> bool:
    """Record chosen parameters, metrics, and importance evidence in MLflow."""
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{(output_dir / 'mlflow.db').resolve().as_posix()}")
        mlflow.set_experiment("w3d4_svm_knn")
        with mlflow.start_run(run_name="svm-vs-knn"):
            for name, result in results.items():
                mlflow.log_params({f"{name}_{key}": value for key, value in result["parameters"].items()})
                mlflow.log_metrics({f"{name}_{key}": value for key, value in result["metrics"].items()})
                mlflow.log_metric(f"{name}_best_cv_roc_auc", result["best_cv_roc_auc"])
            for artifact in artifacts:
                mlflow.log_artifact(str(artifact), artifact_path="evidence")
    except Exception as error:
        warnings.warn(f"MLflow tracking was skipped: {error}", RuntimeWarning, stacklevel=2)
        return False
    return True


def run_experiment(output_dir: Path | str = DEFAULT_OUTPUT_DIR, track_mlflow: bool = True) -> dict[str, Any]:
    """Tune both classifiers, evaluate once on holdout, and save portable evidence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_split()
    results: dict[str, Any] = {}
    searches = model_searches()
    for name, (pipeline, grid) in searches.items():
        search = GridSearchCV(pipeline, grid, scoring="roc_auc", cv=5, n_jobs=1)
        search.fit(X_train, y_train)
        results[name] = {
            "parameters": search.best_params_,
            "best_cv_roc_auc": float(search.best_score_),
            "metrics": evaluate(search.best_estimator_, X_train, X_test, y_train, y_test),
        }

    # Permutation importance measures the selected model's effect on holdout ROC-AUC.
    winner = max(results, key=lambda name: results[name]["metrics"]["test_roc_auc"])
    best_model = searches[winner][0].set_params(**results[winner]["parameters"])
    best_model.fit(X_train, y_train)
    importance_by_metric = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring={"roc_auc": "roc_auc"},
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    roc_auc_importance = importance_by_metric["roc_auc"]
    importance_frame = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": roc_auc_importance["importances_mean"],
            "importance_std": roc_auc_importance["importances_std"],
        }
    ).sort_values("importance_mean", ascending=False)
    importance_path = output_dir / "permutation_importance.csv"
    importance_frame.to_csv(importance_path, index=False)
    figure, axis = plt.subplots(figsize=(8, 5))
    importance_frame.head(10).sort_values("importance_mean").plot.barh(x="feature", y="importance_mean", xerr="importance_std", ax=axis, legend=False, color="#3976af")
    axis.set_title(f"Top permutation importances ({winner.upper()})")
    axis.set_xlabel("Decrease in held-out ROC-AUC")
    figure.tight_layout()
    plot_path = output_dir / "permutation_importance.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)

    summary = {"dataset": "sklearn breast cancer", "random_state": RANDOM_STATE, "holdout_fraction": 0.2, "results": results, "importance_model": winner, "top_features": importance_frame.head(10).to_dict(orient="records")}
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if track_mlflow:
        log_to_mlflow(output_dir, results, (metrics_path, importance_path, plot_path))
    return summary


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))


