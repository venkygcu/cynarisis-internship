from pathlib import Path

import pytest

from src.w3d4_svm_knn import evaluate, load_split, run_experiment


def test_holdout_is_stratified_and_training_only() -> None:
    X_train, X_test, y_train, y_test = load_split()
    assert len(X_train) + len(X_test) == 569
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.02)
    assert set(X_train.index).isdisjoint(X_test.index)


def test_tuned_classifiers_save_metrics_and_importance(tmp_path: Path) -> None:
    results = run_experiment(tmp_path, track_mlflow=False)
    assert set(results["results"]) == {"svm", "knn"}
    for model in results["results"].values():
        assert 0 <= model["metrics"]["test_roc_auc"] <= 1
        assert model["best_cv_roc_auc"] > 0.9
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "permutation_importance.csv").is_file()
    assert (tmp_path / "permutation_importance.png").is_file()
