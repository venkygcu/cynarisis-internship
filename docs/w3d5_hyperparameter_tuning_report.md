# W3D5 — Hyperparameter Tuning: Grid Search and Random Search

This experiment compares exhaustive `GridSearchCV` with a ten-draw
`RandomizedSearchCV` for SVM and KNN on the scikit-learn breast-cancer data.
Both classifiers use `StandardScaler` inside the pipeline, and each search uses
five-fold ROC-AUC cross-validation on the training split. A stratified 80/20
holdout is kept out of model selection for the final accuracy, F1, and ROC-AUC
comparison.

## Choosing a model

Grid search evaluates every listed candidate (15 SVM and 28 KNN settings).
Randomized search samples ten candidates for each algorithm from the defined
search spaces. The recommendation in `metrics.json` is the experiment with the
highest training-fold CV ROC-AUC; the holdout metrics describe its final
performance without choosing the winner from the holdout. This gives a
repeatable way to balance search coverage, compute budget, and validation
performance for the problem.

## Evidence and MLOps

`outputs/w3d5_hyperparameter_tuning/` contains the selected settings and
metrics, every candidate's cross-validation result, a comparison CSV and plot,
and a local MLflow tracking database. The portable JSON, CSV, and PNG evidence
is checked in; the local SQLite database is regenerable.

MLflow records experiment parameters, CV/holdout metrics, and artifacts.
Reproducible seeds, fold-based model selection, and a separate holdout cover the
MLOps practices relevant to this model-selection task. CrewAI and LangGraph
orchestrate agent workflows, and Ragas evaluates generated retrieval answers;
there is no agent or retrieval-generated output in this experiment to use those
components for.

## Run

```powershell
.venv\Scripts\python.exe src\w3d5_hyperparameter_tuning.py
.venv\Scripts\python.exe -m pytest tests\test_w3d5_hyperparameter_tuning.py -q
```

## Self-review checklist

- [x] All preprocessing is fit inside CV training folds.
- [x] Grid and randomized searches cover both SVM and KNN.
- [x] Search method/model recommendation uses CV, not holdout performance.
- [x] Holdout metrics and full CV search results are saved.
- [x] MLflow tracking and portable comparison evidence are generated.
- [ ] CIA Full Stack Mentor review: unavailable in this environment; two mentor interactions remain to be completed and logged.
- [x] Two descriptive commits and push to `feat/aiml-W3-venky`.

