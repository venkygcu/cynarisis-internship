# Week 2 Project: End-to-End Preprocessing Pipeline

## Deliverable

`src/w2_preprocessing_pipeline.py` trains a leakage-safe Titanic classifier:

1. validates the incoming schema and target;
2. stratifies a train/test split before fitting transformers;
3. median-imputes and standardizes numeric features;
4. most-frequent-imputes and one-hot encodes categorical features;
5. persists the fitted `Pipeline`, transformed feature list, quality report, and validation metrics; and
6. records parameters, metrics, and the quality artifact in a local MLflow SQLite experiment.

Run it from the repository root with:

```powershell
.\.venv\Scripts\python.exe src\w2_preprocessing_pipeline.py
```

## Output evidence

The checked-in evidence in `outputs/w2_preprocessing/` was generated from the bundled 891-row Titanic source:

| Check | Result |
| --- | ---: |
| Train / test rows | 712 / 179 |
| Transformed features | 18 |
| Train / test positive rate | 0.3834 / 0.3855 |
| Holdout accuracy | 0.8324 |
| Holdout F1 | 0.7727 |
| Holdout ROC-AUC | 0.8694 |

The generated MLflow database is deliberately ignored because it is a local, regenerable tracking store. The portable JSON, CSV, and fitted model are versioned as submission evidence.

## Approved-stack alignment

MLflow and MLOps practices are active in this deterministic data-preparation task: reproducible seeds, schema validation, leakage-safe fitting, tests, tracked metrics, and portable artifacts. CrewAI, LangGraph, and Ragas are not invoked because this workflow contains neither an LLM agent decision nor retrieval-generated output to orchestrate or evaluate; adding an LLM call would make the preprocessing less deterministic without improving it. Their appropriate follow-on role is to orchestrate data-quality remediation (CrewAI/LangGraph) and score any generated data documentation or RAG assistant (Ragas).

## CIA — Full Stack Mentor Mode interaction log

| Interaction | Review focus | Outcome |
| --- | --- | --- |
| 1 | Correctness, schema boundaries, MLflow persistence, and test coverage | **Request changes:** the validator allowed a null target, which would fail later at `astype(int)`. |
| 2 | Re-review after the null-target guard and regression test | **Approved:** no P0/P1 findings; leakage controls, artifact outputs, and local MLflow tracking were verified. |

The first review finding was resolved before the initial commit by raising a clear validation error for missing targets and adding `test_validate_raw_data_rejects_missing_target`.

## Self-review checklist

- [x] Uses a reproducible, local source dataset.
- [x] Validates schema and binary non-null target before training.
- [x] Fits imputers, scaler, and encoder on training data only.
- [x] Uses an sklearn `Pipeline` to prevent inference/training drift.
- [x] Logs local MLflow parameters, metrics, and artifact evidence.
- [x] Saves model, quality report, validation summary, and feature names.
- [x] Includes automated tests for transformation, output evidence, and invalid inputs.
- [x] Records two CIA Full Stack Mentor-mode review interactions.
