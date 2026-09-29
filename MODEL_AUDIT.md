# Model Audit

## Executive conclusion

The original external-test claim is not valid because all 4,521 rows in `test_y.csv` are also present in the 45,211-row `train.csv`. The current result can still describe how the saved pipeline behaves on those records, but it must not be labeled unseen-data performance.

The existing workflow also uses `duration`, which is recorded after a marketing call. A model that includes it cannot support the stated pre-call targeting decision.

## Verified data checks

| Check | Result |
|---|---:|
| Training rows | 45,211 |
| Positive target rate | 11.70% |
| Rows in `test_y.csv` | 4,521 |
| `test_y.csv` rows found in `train.csv` | 4,521 |
| Overlap | 100% |

## Revised experiment design

- Source: `train.csv`
- Split: stratified 60% training, 20% validation, 20% test
- Random seed: 42
- Preprocessing: numeric standardization and one-hot encoding fitted on the training split
- Estimator: L2-regularized, class-weighted logistic regression
- Threshold: selected on validation data to maximize positive-class F1
- Final test set: used once after threshold selection

## Results

### Pre-call model

The pre-call model excludes `duration`.

| Metric | Threshold 0.5 | Validation-selected threshold 0.6465 |
|---|---:|---:|
| Accuracy | 0.765 | 0.875 |
| Precision, Yes | 0.273 | 0.462 |
| Recall, Yes | 0.607 | 0.417 |
| F1, Yes | 0.377 | 0.438 |
| ROC AUC | 0.763 | 0.763 |

The threshold is a business choice. A lower threshold captures more potential subscribers but sends more low-probability leads to agents. A higher threshold improves precision and reduces call volume at the cost of more false negatives.

### Post-call diagnostic model

The diagnostic model includes `duration` and reaches ROC AUC 0.910 and F1 0.585 at the validation-selected threshold. This confirms that call duration is strongly predictive, but it does not make the feature appropriate for pre-call ranking.

## SMOTE assessment

The presentation compares original and SMOTE workflows mainly with accuracy, ROC AUC, and overall F1. For an imbalanced campaign target, the audit recommends:

1. Put preprocessing and SMOTE inside a single cross-validation pipeline.
2. Resample training folds only.
3. Report positive-class precision, recall, F1, PR AUC, and calibration.
4. Compare SMOTE with class weighting and probability-threshold tuning.
5. Evaluate on an untouched holdout or, preferably, a later campaign period.

Synthetic oversampling is useful only if it improves the business objective on untouched data. Similar accuracy before and after SMOTE does not establish that the resampled model is better.

## Required fixes before publication

- Do not describe `test_y.csv` as unseen external data.
- Separate pre-call and post-call modeling objectives.
- Publish a single reproducible pipeline instead of multiple conflicting notebook branches.
- Add PR AUC, calibration, lift by decile, and a campaign-cost simulation.
- Tune LightGBM and the decision threshold using validation folds only.
- Save the feature schema, training date, data hash, and package versions with the model.

