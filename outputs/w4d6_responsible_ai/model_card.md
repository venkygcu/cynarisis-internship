# Model Card ? Adult Income Logistic Regression

**Version/owner:** Week 4 educational audit / Student. **Seed:** 42.

**Model details:** Logistic Regression with class balancing; numeric median imputation and scaling; categorical mode imputation and one-hot encoding. Gender is excluded from predictors and retained for audit only.

**Intended use:** Educational fairness and explainability demonstration on UCI Adult Census Income. Not intended for hiring, lending, benefits, or individual economic decisions.

**Factors and data:** Historical US census data includes age, education, occupation, hours, and demographics; labels may encode structural inequity. Stratified 75/25 train/test split; test size 12211. Preprocessing fit on training data only.

**Metrics:** Accuracy 0.801; positive-class precision 0.557; recall 0.832; ROC-AUC 0.902; test disparate impact (unprivileged/privileged selection rate) 0.335. Reweighed test DI: 0.587.

**Ethical considerations and caveats:** The old US dataset is not representative of present-day or Indian populations. Binary sex coding excludes gender diversity, and aggregate metrics hide intersectional harms. Demographic parity, equalized odds, and individual fairness encode different goals. SHAP describes model behavior, not causation. Reweighing does not establish fairness. Do not use for consequential decisions; require independent subgroup evaluation, privacy and domain review, monitoring, meaningful human review, and appeal mechanisms.