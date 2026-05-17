# Model Interpretation Notes

## Goal

The goal of this step is to understand which features are driving the baseline models and whether the model behavior looks reasonable.

Previous results show that Logistic Regression, Random Forest, and XGBoost all outperformed the Dummy Classifier. This portion focuses on interpreting those models and identifying which features appear most important.

## Model Results Summary

The first baseline model comparison was:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy Classifier | 0.750 | 0.750 | 1.000 | 0.857 | 0.500 |
| Logistic Regression | 0.825 | 0.960 | 0.800 | 0.873 | 0.916 |
| Random Forest | 0.858 | 0.974 | 0.833 | 0.898 | 0.904 |
| XGBoost | 0.850 | 0.900 | 0.900 | 0.900 | 0.894 |

Logistic Regression achieved the highest ROC-AUC, Random Forest achieved the highest accuracy, and XGBoost achieved the most balanced precision and recall.

Because the current dataset is small and intentionally status-stratified, these results should be treated as early pipeline validation rather than final model performance.

## Feature Importance Outputs

The modeling script saves feature importance outputs to:

- `reports/logistic_regression_feature_importance.csv`
- `reports/random_forest_feature_importance.csv`
- `reports/xgboost_feature_importance.csv`

Logistic Regression feature importance is based on model coefficients. Positive coefficients push the model toward predicting higher risk, while negative coefficients push the model toward predicting lower risk.

Random Forest and XGBoost feature importance are based on tree-based importance scores. These scores show which features were useful to the model, but they do not directly show whether a feature increases or decreases predicted risk.

## Logistic Regression Interpretation

Logistic Regression is the easiest model to interpret because each transformed feature has a coefficient.

Top Logistic Regression features by absolute coefficient:

| Feature | Direction | Notes |
|---|---|---|
| `enrollment_type_ESTIMATED` | Higher risk | Strongest positive coefficient. Estimated enrollment appears strongly associated with higher-risk outcomes. |
| `log_enrollment_count` | Lower risk | Higher log enrollment count is associated with lower predicted risk. |
| `enrollment_type_MISSING` | Lower risk | Missing enrollment type is associated with lower predicted risk in this sample. |
| `enrollment_type_ACTUAL` | Lower risk | Actual enrollment is associated with lower predicted risk. |
| `phase_PHASE1` | Lower risk | Phase 1 trials are associated with lower predicted risk in this sample. |
| `phase_PHASE1; PHASE2` | Lower risk | Combined Phase 1 and Phase 2 trials are associated with lower predicted risk. |
| `sponsor_class_OTHER` | Lower risk | Sponsor class `OTHER` is associated with lower predicted risk in this sample. |
| `enrollment_count` | Higher risk | Raw enrollment count has a positive coefficient, while log enrollment count has a negative coefficient, suggesting the relationship may not be straightforward. |
| `sponsor_class_OTHER_GOV` | Higher risk | Other government sponsors are associated with higher predicted risk in this sample. |
| `phase_PHASE3` | Higher risk | Phase 3 trials are associated with higher predicted risk in this sample. |

Initial interpretation:

- The strongest Logistic Regression signal comes from enrollment-related features, especially `enrollment_type_ESTIMATED`, `enrollment_type_ACTUAL`, and `log_enrollment_count`.
- `enrollment_type_ESTIMATED` pushes predictions toward higher risk, while `enrollment_type_ACTUAL` pushes predictions toward lower risk.
- `log_enrollment_count` has a strong negative coefficient, suggesting that larger log-scaled enrollment counts may be associated with lower predicted risk in this sample.
- Phase-related features also appear important. Phase 1 and Phase 1/2 trials are associated with lower predicted risk, while Phase 2, Phase 3, and Phase 4 appear later among higher-risk coefficients.
- The dominance of enrollment-related features is useful but potentially suspicious. Enrollment information may be updated over the life of a trial, so these fields should be reviewed for leakage risk before treating the model as a strict pre-outcome predictor.

## Random Forest Interpretation

Random Forest feature importance shows which transformed features were most useful for splitting across the trees.

Top Random Forest features:

| Feature | Notes |
|---|---|
| `log_enrollment_count` | Most important Random Forest feature. |
| `enrollment_count` | Second most important feature, confirming a strong enrollment-size signal. |
| `enrollment_type_ESTIMATED` | Important categorical enrollment feature. |
| `enrollment_type_ACTUAL` | Important categorical enrollment feature. |
| `eligibility_text_length` | Text length appears useful as a rough proxy for trial complexity. |
| `start_year` | Trial start timing appears useful to the model. |
| `brief_summary_length` | Summary length appears to contain some predictive signal. |
| `num_locations` | Geographic or operational scale may matter. |
| `phase_PHASE2` | Phase 2 appears as one of the more useful phase categories. |
| `num_countries` | International or multi-country scope may add signal. |

Initial interpretation:

- Random Forest relies heavily on enrollment-related features, especially `log_enrollment_count`, `enrollment_count`, and `enrollment_type`.
- The model also uses `eligibility_text_length`, `brief_summary_length`, and `start_year`, suggesting that trial text length and timing contain some predictive signal.
- Location and country counts appear in the top features, which may reflect trial complexity or operational scope.
- Compared with Logistic Regression, Random Forest agrees that enrollment-related variables are among the strongest predictors.
- Because Random Forest importances do not show direction, these results only indicate which features are useful, not whether they increase or decrease predicted risk.

## XGBoost Interpretation

XGBoost feature importance also shows which transformed features were useful for tree-based prediction.

Top XGBoost features:

| Feature | Notes |
|---|---|
| `enrollment_type_ESTIMATED` | Most important XGBoost feature. |
| `log_enrollment_count` | Strong enrollment-size signal. |
| `enrollment_type_ACTUAL` | Important enrollment type signal. |
| `enrollment_count` | Raw enrollment count remains important. |
| `phase_missing` | Missing phase appears useful to the model. |
| `phase_MISSING` | Missing phase category also appears important after one-hot encoding. |
| `sponsor_class_OTHER` | Sponsor class contributes to XGBoost predictions. |
| `eligibility_text_length` | Eligibility text length contributes some signal. |
| `start_year` | Start year appears useful to the model. |
| `num_countries` | Number of countries appears among the top features. |

Initial interpretation:

- XGBoost also relies strongly on enrollment-related features, especially `enrollment_type_ESTIMATED`, `log_enrollment_count`, and `enrollment_type_ACTUAL`.
- `phase_missing` and `phase_MISSING` both appear among the important features, suggesting that missing phase information carries predictive signal.
- XGBoost uses a broader mix of features than Random Forest, including enrollment, phase missingness, sponsor class, text length, start year, countries, and intervention counts.
- As with Random Forest, XGBoost importances show feature usefulness but not direction.

## Cross-Model Takeaways

Across Logistic Regression, Random Forest, and XGBoost, enrollment-related features consistently appear near the top. This includes `enrollment_type_ESTIMATED`, `enrollment_type_ACTUAL`, `enrollment_count`, and `log_enrollment_count`.

This consistency suggests that enrollment information contains strong predictive signal in the current sample. However, it also raises a leakage concern because enrollment fields may be updated during or after a trial. If the goal is strict pre-launch prediction, these features should be reviewed carefully or tested in a separate model without enrollment-related fields.

Text length features such as `eligibility_text_length` and `brief_summary_length` appear in the tree-based models, suggesting that trial text may contain useful signal even before deeper NLP methods are used.

Phase-related features, especially `phase_missing` and one-hot encoded phase categories, also appear in the importance outputs. This supports the earlier EDA finding that missingness is not just a data quality issue; missing values may carry information about study type or trial design.

Location and country count features appear in the tree-based models, which may reflect trial complexity or geographic scope.

Overall, the feature importance results suggest that the models are learning from plausible trial metadata, but the strongest predictors need leakage review before the results can be interpreted as a realistic pre-outcome risk model.

## Leakage Review

The first modeling pass intentionally excluded:

- `completion_date`
- `primary_completion_date`
- `trial_duration_days`

These fields may introduce target leakage because they are related to information that may only be known after or near the end of a trial.

The strongest feature importance signals come from enrollment-related fields. This is useful but also potentially risky. `enrollment_count` and `enrollment_type` may not always represent information available at trial launch. For example, completed trials may be more likely to have actual enrollment values, while disrupted or incomplete trials may retain estimated enrollment values.

Because of this, future modeling should test a stricter feature set that excludes `enrollment_type` and possibly enrollment-related fields to see whether model performance remains strong without potential leakage.

## Limitations

- The dataset currently contains only 600 records.
- The sample is intentionally status-stratified and does not represent real-world clinical trial outcome rates.
- The target label is simplified.
- Feature importance may not be stable on such a small sample.
- One-hot encoded categorical features can make interpretation harder because one original column becomes many transformed columns.
- Tree-based importances show feature usefulness but not direction.
- Enrollment-related fields may contain leakage depending on when they were last updated.
- The current model has not yet been validated on a larger or more realistic dataset.

## Next Steps

- Pull a larger dataset after confirming the modeling pipeline is stable.
- Re-run ingestion, normalization, feature engineering, modeling, and interpretation on the larger dataset.
- Compare whether feature importance remains stable.
- Test a stricter no-enrollment or reduced-enrollment feature set to check whether model performance depends too heavily on potentially leaky fields.
- Add model interpretation summaries to the README.
- Build a dashboard for data quality, model results, and trial-level exploration.
- Consider a PyTorch text model later to test whether eligibility criteria and brief summaries provide additional signal.