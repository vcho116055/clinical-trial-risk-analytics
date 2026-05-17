# No-Enrollment Model Notes

## Goal

The goal of this step was to test whether the model still performs well after removing enrollment-related fields that may introduce leakage.

Removed features:

- `enrollment_count`
- `log_enrollment_count`
- `enrollment_type`

## Reason

Previous feature importance results showed that enrollment-related features were among the strongest predictors across Logistic Regression, Random Forest, and XGBoost.

This may reflect real signal, but it may also reflect leakage because enrollment fields can be updated during or after a trial.

## Results

Removing enrollment-related features caused model performance to drop substantially.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy Classifier | 0.750 | 0.750 | 1.000 | 0.857 | 0.500 |
| Logistic Regression | 0.620 | 0.866 | 0.583 | 0.697 | 0.680 |
| Random Forest | 0.640 | 0.793 | 0.703 | 0.746 | 0.653 |
| XGBoost | 0.728 | 0.751 | 0.953 | 0.840 | 0.653 |

## Observations

- Model performance dropped sharply after removing `enrollment_count`, `log_enrollment_count`, and `enrollment_type`.
- This suggests that enrollment-related fields were carrying a large portion of the predictive signal in the earlier models.
- The drop supports the earlier leakage concern because enrollment fields may be updated during or after a trial.
- Logistic Regression had the highest ROC-AUC in the no-enrollment setting at 0.680, suggesting that some non-enrollment features still contain predictive signal.
- XGBoost maintained high recall at 0.953, but its ROC-AUC dropped to 0.653, suggesting it may be overpredicting the high-risk class rather than cleanly separating high-risk and low-risk trials.
- The no-enrollment models still outperform the dummy baseline on ROC-AUC, but the performance is much weaker than the original larger-sample models.

## Interpretation

The no-enrollment experiment shows that the original model performance should be interpreted cautiously. Enrollment-related fields appear highly predictive, but they may not be safe features for a strict pre-outcome or pre-launch prediction setting.

This does not invalidate the project. Instead, it clarifies that there are two possible modeling setups:

1. A broader metadata model that includes enrollment fields and achieves stronger performance.
2. A stricter pre-outcome model that excludes enrollment fields and has weaker but less leakage-prone performance.

For a strict pre-outcome risk prediction system, the stricter model is more trustworthy, while the broader model is useful for understanding which reported trial metadata is associated with final trial status.

## Feature Importance After Removing Enrollment Features

After removing `enrollment_count`, `log_enrollment_count`, and `enrollment_type`, the models shifted toward trial design, text length, sponsor, phase, time, and location features.

### Logistic Regression

Top Logistic Regression features included:

| Feature | Direction | Interpretation |
|---|---|---|
| `phase_PHASE1; PHASE2` | Higher risk | Combined Phase 1 and Phase 2 trials pushed predictions toward higher risk. |
| `sponsor_class_FED` | Lower risk | Federal sponsor class pushed predictions toward lower risk in this sample. |
| `start_year` | Higher risk | More recent start years pushed predictions toward higher risk. |
| `eligibility_text_length` | Higher risk | Longer eligibility criteria pushed predictions toward higher risk. |
| `sponsor_class_NETWORK` | Lower risk | Network sponsor class pushed predictions toward lower risk. |
| `phase_PHASE1` | Lower risk | Phase 1 trials pushed predictions toward lower risk. |
| `phase_PHASE2` | Higher risk | Phase 2 trials pushed predictions toward higher risk. |
| `sponsor_class_OTHER` | Higher risk | Sponsor class `OTHER` pushed predictions toward higher risk. |
| `phase_PHASE2; PHASE3` | Lower risk | Combined Phase 2 and Phase 3 trials pushed predictions toward lower risk. |
| `num_conditions` | Higher risk | Trials with more listed conditions pushed predictions toward higher risk. |

Logistic Regression had the highest ROC-AUC among the no-enrollment models at 0.680. Its most important features suggest that even without enrollment fields, there is some signal from phase, sponsor class, start year, eligibility criteria length, and condition count.

### Random Forest

Top Random Forest features included:

| Feature | Interpretation |
|---|---|
| `eligibility_text_length` | Most important Random Forest feature after removing enrollment fields. |
| `start_year` | Trial timing was highly important. |
| `brief_summary_length` | Summary text length contributed strongly. |
| `num_locations` | Location count may reflect trial scale or operational complexity. |
| `num_interventions` | Intervention count may reflect trial complexity. |
| `num_conditions` | Condition count contributed to model decisions. |
| `num_countries` | Country count may reflect geographic complexity. |
| `phase_MISSING` | Missing phase carried predictive signal. |
| `sponsor_class_OTHER` | Sponsor class contributed to predictions. |
| `phase_missing` | Missingness itself was useful to the model. |

Random Forest relied heavily on text length and timing features, especially `eligibility_text_length`, `start_year`, and `brief_summary_length`. It also used trial complexity features such as locations, interventions, conditions, and countries.

### XGBoost

Top XGBoost features included:

| Feature | Interpretation |
|---|---|
| `phase_MISSING` | Missing phase was the most important XGBoost feature. |
| `sponsor_class_NETWORK` | Sponsor class contributed to predictions. |
| `phase_missing` | Missing phase indicator was highly useful. |
| `num_countries` | Geographic complexity contributed signal. |
| `start_year` | Trial timing remained important. |
| `sex_MALE` | Male-only eligibility contributed to predictions. |
| `num_locations` | Location count contributed signal. |
| `eligibility_text_length` | Eligibility criteria length remained useful. |
| `has_placebo` | Placebo keyword indicator contributed signal. |
| `sponsor_class_INDUSTRY` | Industry sponsor class contributed to predictions. |

XGBoost relied on a broader mix of phase missingness, sponsor class, geography, timing, sex eligibility, text length, and placebo indicators. Unlike the original larger model, enrollment-related fields no longer dominated the model because they were removed.

## Cross-Model Interpretation

Removing enrollment-related fields caused model performance to drop substantially, but the remaining feature importance outputs are more consistent with a stricter pre-outcome modeling setup.

Across the no-enrollment models, several themes appeared repeatedly:

- Text length features, especially `eligibility_text_length` and `brief_summary_length`, became more important.
- `start_year` remained important, suggesting that timing or reporting patterns may be associated with trial status.
- Phase-related features, especially missing phase indicators, carried predictive signal.
- Sponsor class appeared across Logistic Regression and XGBoost.
- Location and country counts appeared in the tree-based models, suggesting that trial scale or geographic complexity may still matter.

The no-enrollment experiment suggests that the original high-performing models were strongly influenced by enrollment-related fields. After removing those fields, performance became weaker but still remained above the dummy baseline in ROC-AUC. This suggests that the non-enrollment features contain some predictive signal, but not nearly as much as the enrollment-related fields in the current sample.

## Limitations

- The dataset is still status-stratified and does not represent real-world clinical trial outcome rates.
- Removing enrollment features reduces one major leakage concern, but it does not eliminate all possible leakage.
- Some remaining fields, such as `start_year`, sponsor class, and missingness indicators, may reflect reporting patterns rather than causal risk factors.
- Feature importance may change when the pipeline is run on a larger or less artificially balanced dataset.
- The no-enrollment model should be treated as a conservative baseline rather than a final production model.

## Takeaways

The no-enrollment model performs worse but is more conservative from a leakage perspective.

The project now has two useful modeling views:

1. A broader reported-metadata model that includes enrollment fields and achieves stronger performance.
2. A stricter no-enrollment model that reduces leakage risk but performs more modestly.

This makes the project more credible because it does not blindly rely on the highest model score. Instead, it investigates whether the strongest predictors are safe to use for a realistic pre-outcome prediction task.

## Next Steps

- Compare the broader metadata model and no-enrollment model in the README.
- Decide which model version should be shown in the dashboard (or maybe even both).
- Use the no-enrollment model as the more conservative pre-outcome model.
- Use the broader metadata model as an exploratory reported-metadata model.
- Begin dashboard development using the larger processed dataset and saved model reports.