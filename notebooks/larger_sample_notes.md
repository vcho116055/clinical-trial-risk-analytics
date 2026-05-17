# Larger Sample Notes

## Goal

The goal was to test whether the pipeline works successfully after scaling from initial 600-record sample to a larger 2000-record sample. 

## Larger Sample Pull

- Target statuses: `COMPLETED`, `TERMINATED`, `WITHDRAWN`, `SUSPENDED`
- Target records per status: 500
- Expected total records: 2000
- Raw output: `data/raw/clinical_trials_larger_sample.json`

## Outputs 

- Normalized table: `data/interim/trials_normalized_larger_sample.csv`
- Processed modeling dataset: `data/processed/trials_modeling_larger_sample.csv`
- Model results: `reports/model_results_larger.csv`

## Small vs Larger Sample Results 

Small Sample

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy Classifier | 0.750 | 0.750 | 1.000 | 0.857 | 0.500 |
| Logistic Regression | 0.825 | 0.960 | 0.800 | 0.873 | 0.916 |
| Random Forest | 0.858 | 0.974 | 0.833 | 0.898 | 0.904 |
| XGBoost | 0.850 | 0.900 | 0.900 | 0.900 | 0.894 |

Larger Sample

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy Classifier | 0.750 | 0.750 | 1.000 | 0.857 | 0.500 |
| Logistic Regression | 0.840 | 0.968 | 0.813 | 0.884 | 0.910 |
| Random Forest | 0.835 | 0.947 | 0.827 | 0.883 | 0.906 |
| XGBoost | 0.852 | 0.906 | 0.897 | 0.901 | 0.916 |

## Observations

- Model performance remained stable after scaling from the 600-record sample to the larger 2,000-record sample. This suggests that the pipeline and engineered features are not only working on the small prototype dataset.
- Logistic Regression improved slightly in accuracy, precision, recall, and F1, while ROC-AUC decreased only slightly from 0.916 to 0.910.
- Random Forest performed similarly across both samples. Its ROC-AUC increased slightly from 0.904 to 0.906, while accuracy, precision, recall, and F1 decreased slightly.
- XGBoost improved the most on the larger sample. Its ROC-AUC increased from 0.894 to 0.916, and it achieved the highest F1 score on the larger sample.
- On the larger sample, XGBoost appears to be the strongest overall model because it has the highest accuracy, F1, and ROC-AUC, while also keeping precision and recall relatively balanced.
- The Dummy Classifier results stayed exactly the same because both samples are status-stratified with the same 75% high-risk and 25% low-risk class balance.

## Limitations

- The larger sample is still status-stratified 
- The dataset is larger but still not necessarily representative of real-world outcome rates 
- Enrollment-related features still might be affected by leakage. 
- Additional validation is needed before treating any results as final. 

## Next Steps

- Compare feature importance stability between the small and larger samples.
- Consider a stricter model without enrollment-related features.
- Validate the pipeline on a less artificially balanced dataset.
- Begin dashboard development once the larger sample pipeline is stable.