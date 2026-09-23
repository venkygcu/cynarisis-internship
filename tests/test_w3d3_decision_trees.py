from pathlib import Path

import pytest
from sklearn.tree import DecisionTreeClassifier

from src.w3d3_decision_trees import evaluate, gini_impurity, information_gain, load_split, run_experiment


def test_gini_impurity_for_pure_and_balanced_nodes() -> None:
    assert gini_impurity([10, 0]) == 0.0
    assert gini_impurity([5, 5]) == pytest.approx(0.5)


def test_information_gain_rewards_a_pure_split() -> None:
    assert information_gain([5, 5], [5, 0], [0, 5]) == pytest.approx(0.5)


def test_holdout_is_stratified_and_metrics_are_bounded() -> None:
    X_train, X_test, y_train, y_test = load_split()
    metrics = evaluate(DecisionTreeClassifier(max_depth=3, random_state=42), X_train, X_test, y_train, y_test)
    assert len(X_train) + len(X_test) == 569
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.02)
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_experiment_writes_required_evidence(tmp_path: Path) -> None:
    summary = run_experiment(tmp_path, track_mlflow=False)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "tuned_tree.png").exists()
    assert (tmp_path / "forest_feature_importance.png").exists()
    assert summary["tuned_tree"]["test_roc_auc"] > 0.9
