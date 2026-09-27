# W3D4 — SVM & KNN: When to Use What

This exercise compares Support Vector Machines and k-Nearest Neighbors on the
scikit-learn breast-cancer dataset. It uses a reproducible stratified 80/20
holdout and five-fold ROC-AUC grid search. Each model is inside a pipeline with
`StandardScaler`, so scaling is fitted within each training fold and applied
consistently at evaluation time.

## What the comparison shows

SVM can work well when a separating margin is useful; its kernel and `C`
parameters control the boundary and regularization. KNN predicts from nearby
training examples; its neighbor count, distance weighting, and distance metric
control its behavior. Both rely on meaningful feature distances, so scaling is
important. The held-out ROC-AUC, F1, accuracy, and selected cross-validation
score are saved in `outputs/w3d4_svm_knn/metrics.json`.

The top tuned classifier is explained with permutation importance on the
held-out set. This measures the drop in ROC-AUC when a feature is shuffled; it
is an interpretation aid rather than an intrinsic model coefficient. CSV and
PNG evidence are saved beside the metrics.

## Approved stack and MLOps

MLflow records selected hyperparameters, cross-validation scores, holdout
metrics, and the output artifacts in a local SQLite tracking store. Reproducible
splits and tuning plus an evaluation holdout provide the MLOps foundations for
this tabular model comparison. CrewAI and LangGraph are orchestration tools,
and Ragas evaluates generated retrieval responses; this classifier has no
agent or RAG output to orchestrate or evaluate, so those tools are not relevant
to its execution.

## Run

```powershell
.venv\Scripts\python.exe src\w3d4_svm_knn.py
.venv\Scripts\python.exe -m pytest tests\test_w3d4_svm_knn.py -q
```

## Self-review checklist

- [x] Stratified holdout is separate from model selection.
- [x] Scaling is fitted inside each cross-validation pipeline.
- [x] SVM and KNN parameters are tuned with five-fold ROC-AUC.
- [x] Accuracy, F1, ROC-AUC, and CV ROC-AUC are saved for both models.
- [x] Permutation importance and MLflow artifacts are included.
- [ ] CIA Full Stack Mentor review: not completed here; no CIA mentor connector is available in this environment. The required two mentor interactions remain pending.
- [x] Two descriptive commits created and pushed on `feat/aiml-W3-venky`.



