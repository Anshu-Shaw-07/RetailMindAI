# RetailMind AI

RetailMind AI is an end-to-end retail AI platform built incrementally. Each daily
increment adds one module and integrates with the same codebase.

**Current increment: Day 3 — multi-model comparison, tuning, and bias–variance.**

## Repository layout

| Path | Contents |
|---|---|
| `data/Telco-Customer-Churn.csv` | Telco Customer Churn dataset |
| `notebooks/train_churn.ipynb` | Day 1 baseline training |
| `notebooks/predict_churn.ipynb` | Day 1 prediction walkthrough |
| `notebooks/retailmind_churn_models.ipynb` | Day 2 model comparison |
| `notebooks/retailmind_day3_tuning.ipynb` | **Day 3 tuning, validation, and champion lock** |
| `src/services/churn.py` | Shared churn scoring service |
| `models/` | Versioned model artifacts and metrics |

Day 3 does not modify any Day 1 or Day 2 notebook or model file.

## Day 3 — tuned churn champion

**Goal:** tune competing churn models, diagnose underfitting vs overfitting, and
lock a robust champion that generalises to unseen customers.

### Validation protocol

- Clean 7,043 rows; 19 raw features after dropping `customerID`.
- One stratified 80/20 split (`random_state=42`); the 20% holdout is evaluated
  **once**, after selection.
- 5-fold `StratifiedKFold` (`shuffle=True`, `random_state=42`) on the development set.
- `GridSearchCV` with multi-metric scoring; models are ranked by **ROC-AUC**
  (recall breaks ties) because churn is imbalanced and the business needs risk
  ranking, not just a hard label.
- Preprocessing (median/most-frequent imputation, one-hot encoding, scaling) lives
  inside each pipeline, so every fold learns it from its own training data.

### Locked champion

| Item | Value |
|---|---|
| Model | **Random Forest** |
| Artifact | `models/churn_v3.joblib` |
| Hyperparameters | `n_estimators=200`, `max_depth=10`, `min_samples_leaf=5`, `class_weight="balanced"` |
| CV ROC-AUC | **0.8464** ± 0.0093 |
| CV recall | 0.7746 |
| CV F1 | 0.6314 |
| Train–CV ROC-AUC gap | +0.0738 (high variance) |
| Holdout ROC-AUC | **0.8419** |
| Holdout recall | 0.7727 |
| Holdout F1 | 0.6215 |
| Holdout accuracy | 0.7502 |

The saved artifact is a complete scikit-learn `Pipeline` (preprocessing +
classifier) and was verified to reload and reproduce its predictions.

### Cross-validation leaderboard

| Model | Best hyperparameters | CV ROC-AUC | CV std | CV recall | CV F1 | Train ROC-AUC | Train–CV gap | Diagnosis |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **Random Forest** | `max_depth=10`, `min_samples_leaf=5`, `n_estimators=200` | **0.8464** | 0.0093 | 0.7746 | 0.6314 | 0.9202 | 0.0738 | High variance |
| Logistic Regression | `C=10.0` | 0.8462 | 0.0127 | 0.8000 | 0.6275 | 0.8498 | 0.0036 | Balanced |
| SVM (RBF) | `C=1.0`, `gamma=0.01` | 0.8424 | 0.0101 | 0.8254 | 0.6040 | 0.8491 | 0.0068 | Balanced |
| Decision Tree | `max_depth=4`, `min_samples_leaf=1` | 0.8279 | 0.0106 | 0.7920 | 0.6030 | 0.8418 | 0.0139 | Balanced |

All candidates use `class_weight="balanced"`. Logistic Regression and SVM use
standardised numeric features; the tree models do not.

### Bias–variance read

Thresholds used in the notebook: a train–CV gap **> 0.05** is high variance; a CV
ROC-AUC **< 0.70** with a small gap is high bias.

- **Random Forest** wins on CV ROC-AUC but carries a **0.0738 variance gap** —
  it fits the training folds noticeably better than unseen folds. This is the
  main reliability risk.
- **Logistic Regression** is essentially tied (0.8462 vs 0.8464) with a **0.0036
  gap** and higher recall. It is the lower-variance fallback if the variance gap
  becomes a deployment concern.
- **SVM** is balanced and has the highest CV recall (0.8254), but the lowest F1
  among the top three.
- **Decision Tree** is balanced but weaker; shallow depth (4) limits its capacity.

### Learning-curve verdicts

| Model | Verdict |
|---|---|
| Random Forest | Variance gap persists → more regularisation or a simpler model |
| Logistic Regression | Curves converged → diminishing returns from more data |

**Decision:** keep the tuned Random Forest as the Day 3 champion (best CV and
holdout ROC-AUC), and target its variance in Day 4 through feature engineering,
stronger regularisation, and bagging/boosting ensembles rather than assuming the
current feature set is final.

### Holdout results

| Metric | Value |
|---|---:|
| ROC-AUC | 0.8419 |
| Recall | 0.7727 |
| F1 | 0.6215 |
| Precision | 0.5198 |
| Accuracy | 0.7502 |

The holdout ROC-AUC (0.8419) is close to the CV estimate (0.8464), so the
selection process generalised as expected.

### Day 3 artifacts

| File | Contents |
|---|---|
| `models/churn_v3.joblib` | Tuned champion pipeline |
| `models/day3_cv_results.csv` | CV leaderboard for every candidate |
| `models/day3_test_metrics.csv` | One-shot holdout metrics |
| `models/day3_best_params.json` | Champion and per-model hyperparameters |
| `models/day3_model_card.md` | Full model card with CV and bias–variance notes |

### Reproduce

Open and run `notebooks/retailmind_day3_tuning.ipynb` from the repo root or from
`notebooks/`, or execute it headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 \
  notebooks/retailmind_day3_tuning.ipynb
```

Environment used: scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4, Python 3.13.

### Integrating the champion

`src/services/churn.py` accepts a `model_path` argument, so the tuned pipeline can
be scored without code changes:

```python
from src.services.churn import predict

predict(customer_record, model_path="models/churn_v3.joblib")
```

The default path in the service still points to the Day 1 pipeline; promoting
`churn_v3.joblib` to the default is a separate, reviewable change.

## Next iteration (Day 4)

Engineer domain features (tenure bands, service counts, spend ratios), apply PCA
for a compact linear view, and add bagging/boosting ensembles to finalise the
churn engine. Day 4 should reuse this notebook's cross-validation and
bias–variance workflow so improvements are measured rather than assumed.
