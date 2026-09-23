# W3D3 — Decision Trees & Random Forests

This exercise uses the scikit-learn breast-cancer dataset with a reproducible,
stratified 80/20 holdout (`random_state=42`).  The implementation in
`src/w3d3_decision_trees.py` first trains an unrestricted decision tree, then
uses 5-fold ROC-AUC grid search to tune `criterion`, `max_depth`, and
`min_samples_leaf`.  It also tunes a Random Forest baseline.

## Learning evidence

- `gini_impurity` implements \(1 - \sum p_i^2\); a pure node has impurity 0 and
  a 50/50 binary node has impurity 0.5.
- `information_gain` calculates the parent Gini score minus its weighted child
  Gini scores. The splitter selects the highest positive gain.
- Comparing train and held-out accuracy makes the unrestricted tree's
  overfitting observable. Limiting depth and minimum leaf size is selected only
  from training-fold CV, preserving the test set as an honest final check.
- `tuned_tree.png`, `forest_feature_importance.png`, `tree_cv_results.csv`, and
  `metrics.json` are generated in `outputs/w3d3_decision_trees/`.

## Approved stack and MLOps

MLflow records all selected hyperparameters, holdout metrics, and visualization
artifacts to a local, project-scoped tracking directory. This is the applicable
MLOps component for a supervised tabular classification experiment. CrewAI and
LangGraph are agent-orchestration tools, and Ragas evaluates LLM/RAG responses;
they are intentionally not inserted into this non-agentic, non-RAG model
training pipeline. That keeps the experiment valid without claiming irrelevant
evaluation coverage.

## Run

```powershell
.venv\Scripts\python.exe src/w3d3_decision_trees.py
.venv\Scripts\python.exe -m pytest tests/test_w3d3_decision_trees.py -q
```
