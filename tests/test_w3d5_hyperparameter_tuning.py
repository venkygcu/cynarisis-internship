from pathlib import Path

import pytest

from src.w3d5_hyperparameter_tuning import load_split, run_experiment, search_spaces


def test_holdout_is_stratified_and_disjoint() -> None:
    X_train, X_test, y_train, y_test = load_split()
    assert len(X_train) + len(X_test) == 569
    assert set(X_train.index).isdisjoint(X_test.index)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.02)


def test_search_spaces_cover_grid_and_random_svm_knn() -> None:
    spaces = search_spaces()
    assert set(spaces) == {"svm", "knn"}
    assert all({"pipeline", "grid", "random"} <= config.keys() for config in spaces.values())


def test_experiment_saves_comparison_and_uses_cv_for_recommendation(tmp_path: Path) -> None:
    summary = run_experiment(tmp_path, track_mlflow=False)
    assert set(summary["results"]) == {"svm_grid", "svm_random", "knn_grid", "knn_random"}
    assert summary["recommended_experiment"] == max(
        summary["results"], key=lambda name: summary["results"][name]["best_cv_roc_auc"]
    )
    assert summary["results"]["svm_grid"]["candidate_count"] == 15
    assert summary["results"]["svm_random"]["candidate_count"] == 10
    assert summary["results"]["knn_grid"]["candidate_count"] == 28
    for name in summary["results"]:
        assert 0 <= summary["results"][name]["metrics"]["test_roc_auc"] <= 1
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "search_comparison.csv").is_file()
    assert (tmp_path / "cv_results.csv").is_file()
    assert (tmp_path / "search_comparison.png").is_file()
