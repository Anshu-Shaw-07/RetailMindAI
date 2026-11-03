# Day 3 model card - churn_v3

## Champion

- **Model:** Random Forest
- **Best hyperparameters:** {'classifier__max_depth': 10, 'classifier__min_samples_leaf': 5, 'classifier__n_estimators': 200}
- **Probability calibration:** Not needed - champion natively supports predict_proba
- **Saved artifact:** `models/churn_v3.joblib`
- **Dataset:** Telco Customer Churn, 7043 clean rows, 19 raw features
- **CV protocol:** StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- **Selection metric:** ROC-AUC (recall as tie-breaker)

## Cross-validation leaderboard

| Model | Best Parameters | CV ROC-AUC | CV ROC-AUC Std | CV Recall | CV F1 | Train ROC-AUC | Train-CV Gap | Diagnosis |
|---|---|---|---|---|---|---|---|---|
| Random Forest | {'classifier__max_depth': 10, 'classifier__min_samples_leaf': 5, 'classifier__n_estimators': 200} | 0.8464 | 0.0093 | 0.7746 | 0.6314 | 0.9202 | 0.0738 | High variance (overfitting) |
| Logistic Regression | {'classifier__C': 10.0} | 0.8462 | 0.0127 | 0.8 | 0.6275 | 0.8498 | 0.0036 | Balanced |
| SVM (RBF) | {'classifier__C': 1.0, 'classifier__gamma': 0.01} | 0.8424 | 0.0101 | 0.8254 | 0.604 | 0.8491 | 0.0068 | Balanced |
| Decision Tree | {'classifier__max_depth': 4, 'classifier__min_samples_leaf': 1} | 0.8279 | 0.0106 | 0.792 | 0.603 | 0.8418 | 0.0139 | Balanced |

## Holdout test set (20% stratified, evaluated once)

- **Model:** Random Forest
- **Accuracy:** 0.7502
- **Precision:** 0.5198
- **Recall:** 0.7727
- **F1:** 0.6215
- **ROC-AUC:** 0.8419

## Bias-variance diagnosis

              Model                   Diagnosis
      Random Forest High variance (overfitting)
Logistic Regression                    Balanced
          SVM (RBF)                    Balanced
      Decision Tree                    Balanced

## Learning-curve verdicts

- **Random Forest:** Variance gap persists -> more regularisation or simpler model
- **Logistic Regression:** Curves converged -> diminishing returns from more data

## Reproduce

Run `notebooks/retailmind_day3_tuning.ipynb` from the repo root or from `notebooks/`.
The test split is fixed with `random_state=42`, so the holdout numbers are reproducible.
