# Modeling Notes

## Goal

The goal of this modeling step is to test whether the engineered pre-outcome clinical trial features contain predictive signal for the initial binary risk label.

Day 1 focused on pulling and normalizing data. Day 2 focused on EDA and feature engineering. Day 3 focuses on training baseline classification models and comparing them against a dummy baseline.

## Dataset

Input dataset:

data/processed/trials_modeling_sample.csv

The dataset contains 600 clinical trial records and 25 processed columns.

The target variable is `risk_label`:

0 = low risk, 1 = high risk

The current target mapping:
- Low risk = COMPLETED
- High risk = TERMINATED, WITHDRAWN, SUSPENDED

The dataset is intentionally status-stratified, so these results should be treated as preliminary. The current sample is useful for proving the modeling workflow, but it should not be interpreted as the real-world distribution of clinical trial outcomes.

## Feature Set

The first modeling pass uses structured features that would plausibly be available before or near trial launch.

Numeric features: 
- `enrollment_count`
- `log_enrollment_count`
- `num_conditions`
- `num_interventions`
- `num_locations`
- `num_countries`
- `eligibility_text_length`
- `brief_summary_length`
- `start_year`
- `has_placebo`
- `adult_only`
- `phase_missing`
- `maximum_age_missing`
- `locations_missing`
- `interventions_missing`

Categorical features:
- `phase`
- `study_type`
- `sponsor_class`
- `sex`
- `enrollment_type`

Completion-related fields such as `completion_date`, `primary_completion_date`, and `trial_duration_days` were excluded from the first modeling pass because they may introduce target leakage. If the goal is to predict trial risk before the outcome is known, completion-based fields should not be used as model inputs.

## Preprocessing 

Numeric features were processed with median imputation and standard scaling. 

Categorical features were processed with missing value imputation using "MISSING" and one-hot encoding. 

The preprocessing was fit only on the training set, then applied to the test set. This avoids leaking information from the test set into the training process. 

## Train/Test Split

The dataset was split into training and test sets using `test_size = 0.2`, `random_state = 42`, `stratify = y`. 

Stratification keeps the high-risk and low-risk label proportions similar across the training and test sets. 

## Models Trained

The following models were trained: 
- Dummy Classifier
- Logistic Regression
- Random Forest
- XGBoost

The Dummy Classifier is used as a weak baseline. It predicts the most frequent class and gives a reference point for whether real models are learning meaningful signal.

## Model Comparison

| Model               | Accuracy | Precision | Recall |    F1 | ROC-AUC |
| ------------------- | -------: | --------: | -----: | ----: | ------: |
| Dummy Classifier    |    0.750 |     0.750 |  1.000 | 0.857 |   0.500 |
| Logistic Regression |    0.825 |     0.960 |  0.800 | 0.873 |   0.916 |
| Random Forest       |    0.858 |     0.974 |  0.833 | 0.898 |   0.904 |
| XGBoost             |    0.850 |     0.900 |  0.900 | 0.900 |   0.894 |

## Initial Findings 

All three real models greatly outperformed the Dummy Classifier, especially on ROC-AUC. The Dummy Classifier achieved 75% accuracy since the dataset contains more high-risk records than low-risk records, but its ROC-AUC was 0.500 meaning it had no ability to separate high-risk and low-risk trials. 

The Logistic Regression achieved the highest ROC-AUC at 0.916, suggesting that the engineered features contain strong linear predictive signal. The Random Forest achieved the highest accuracy at 0.858 and a strong F1 score of 0.898. It performed slightly better than Logistic Regression on accuracy and F1, but slightly worse on ROC-AUC. 

XGBoost achieved the highest F1 score at 0.900 and the most balanced precision and recall, both at 0.900. This suggests that XGBoost may be useful when balancing false positives and false negatives is more important than maximizing a single metric. 

Overall, the first modeling pass suggests that the engineered pre-outcome features contain meaningful predictive signal. However, the results are preliminary because the current dataset is small and intentionally sampled by status. 

## Metric Interpretation

Accuracy alone is not sufficient for this problem because the target is imbalanced. A model can achieve high accuracy by overpredicting the high-risk class. 

Precision addresses the accuracy of the model when it predicts high risk, while recall answers how many of the actual high-risk trials the model catches. F1 balances the two, and ROC-AUC measures how well the model separates high-risk and low-risk trials across different classification thresholds. 

For this project, recall is crucial since missing a genuinely high-risk trial could be costly. However, precision matters because too many false risk alerts would reduce trust in the model. 

## Limitations

- The current sample has only 600 records.
- The sample was intentionally stratified by status and does not represent real-world clinical trial outcome rates.
- The target label is simplified and only uses trial status.
- The model does not yet directly predict recruitment difficulty.
- Text fields are currently represented through simple length and keyword features, not deep NLP.
- Model performance has not yet been tested on a larger dataset.
- The current train/test split is useful for a first pass, but future work should consider more robust validation.

## Next Steps

- Pull a larger dataset after confirming the modeling pipeline is stable.
- Re-run ingestion, normalization, feature engineering, and modeling on the larger dataset.
- Add model interpretation using feature importance.
- Compare which features contribute most across Logistic Regression, Random Forest, and XGBoost.
- Consider stricter leakage checks for fields that may be updated after trial completion.
- Build a dashboard for data quality, model results, and trial-level exploration.
- Consider adding a PyTorch text model later to test whether eligibility criteria and brief summaries contain additional predictive signal.