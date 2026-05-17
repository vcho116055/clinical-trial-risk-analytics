# Clinical Trial Risk and Enrollment Analytics Platform

An end to end data science project that uses public ClinicalTrials.gov records to analyze clinical trial risk, build baseline prediction models, and prepare for an interactive dashboard that explores risk patterns, model outputs, and data quality.

## Project Overview

Clinical trials can terminate early, withdraw before enrollment, or become suspended for many reasons, including recruitment difficulty, study complexity, eligibility restrictions, sponsor constraints, geography, and trial design.

This project builds a reproducible clinical trial analytics pipeline that moves from raw public API data to a modeling ready dataset, baseline machine learning models, and model interpretation outputs.

The current goal is not to produce clinical or operational recommendations. The goal is to build a transparent, explainable, and reproducible data science workflow for exploring trial risk signals.

## Data Source

Primary data source:

```text
ClinicalTrials.gov public study records
```

The initial dataset is pulled from the ClinicalTrials.gov API and saved as raw JSON before being normalized into flat tables for analysis and modeling.

## Target Variable

The first version uses trial status as a simplified proxy for risk.

Initial target mapping:

| Trial Status | Risk Label | Meaning |
|---|---:|---|
| `COMPLETED` | 0 | Low risk |
| `TERMINATED` | 1 | High risk |
| `WITHDRAWN` | 1 | High risk |
| `SUSPENDED` | 1 | High risk |

Other statuses, such as recruiting, not yet recruiting, active not recruiting, unknown, and other ongoing statuses are excluded or held out for later modeling decisions.

This target is intentionally simple for the first version. It does not capture every form of clinical trial difficulty, such as slow recruitment, delayed completion, poor reporting, or inconclusive outcomes.

## Pipeline Overview

The current pipeline follows this structure:

1. Pull sample records from ClinicalTrials.gov
2. Save raw API output as JSON
3. Normalize nested trial records into a flat table
4. Perform exploratory data analysis
5. Create a processed modeling dataset
6. Train baseline classification models
7. Save model results and feature importance outputs
8. Document limitations and next steps

## Project Structure

```text
data/
  raw/            Raw ClinicalTrials.gov API outputs
  interim/        Normalized flat trial tables
  processed/      Modeling ready datasets

notebooks/
  data_dictionary_notes.md
  initial_eda.py
  day3_modeling_notes.md
  day4_model_interpretation_notes.md

src/
  clinical_trial_risk/
    ingest/       API data pulling
    clean/        Raw JSON normalization
    features/     Feature engineering
    models/       Baseline modeling and interpretation
    dashboard/    Planned Streamlit dashboard

reports/
  model_results.csv
  logistic_regression_feature_importance.csv
  random_forest_feature_importance.csv
  xgboost_feature_importance.csv
```

## Feature Engineering

The first processed modeling dataset includes structured, text derived, and missingness based features.

Examples include:

| Feature | Description |
|---|---|
| `enrollment_count` | Reported trial enrollment count |
| `log_enrollment_count` | Log transformed enrollment count |
| `num_conditions` | Number of listed conditions |
| `num_interventions` | Number of listed interventions |
| `num_locations` | Number of listed study locations |
| `num_countries` | Number of unique countries |
| `eligibility_text_length` | Character length of eligibility criteria |
| `brief_summary_length` | Character length of brief summary |
| `start_year` | Parsed trial start year |
| `has_placebo` | Keyword indicator for placebo |
| `adult_only` | Indicator for adult or older adult only studies |
| `phase_missing` | Indicator for missing phase |
| `maximum_age_missing` | Indicator for missing maximum age |
| `locations_missing` | Indicator for missing location text |
| `interventions_missing` | Indicator for missing intervention text |

Categorical features include:

```text
phase
study_type
sponsor_class
sex
enrollment_type
```

Completion related fields such as `completion_date`, `primary_completion_date`, and `trial_duration_days` are excluded from the first modeling pass because they may leak information about the final trial outcome.

## Exploratory Data Analysis

The initial EDA focuses on:

```text
dataset structure
target distribution
missingness
missingness by status
categorical feature distributions
numeric feature distributions
text length features
date fields
feature readiness
leakage risks
```

Main EDA takeaways:

- The normalized sample contains 600 clinical trial records.
- The sample is intentionally status stratified, so it should not be interpreted as the real world distribution of trial outcomes.
- Missingness is manageable, but some missing values may carry meaning.
- `phase` and `maximum_age` require explicit missing value handling.
- Text fields such as eligibility criteria and brief summaries are useful candidates for simple text length features and later NLP modeling.
- Completion based fields may introduce target leakage and should be handled carefully.

## Modeling

The first baseline modeling pipeline compares:

1. Dummy Classifier
2. Logistic Regression
3. Random Forest
4. XGBoost

The models use pre outcome trial features such as phase, study type, sponsor class, enrollment, condition counts, intervention counts, location counts, text length features, keyword indicators, and missingness indicators.

## Modeling Results

Current baseline results:

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Dummy Classifier | 0.750 | 0.750 | 1.000 | 0.857 | 0.500 |
| Logistic Regression | 0.825 | 0.960 | 0.800 | 0.873 | 0.916 |
| Random Forest | 0.858 | 0.974 | 0.833 | 0.898 | 0.904 |
| XGBoost | 0.850 | 0.900 | 0.900 | 0.900 | 0.894 |

Initial interpretation:

- All three real models substantially outperform the Dummy Classifier.
- Logistic Regression achieves the highest ROC AUC.
- Random Forest achieves the highest accuracy.
- XGBoost achieves the most balanced precision and recall.
- The results suggest that the engineered features contain predictive signal.
- These results are preliminary because the current dataset is small and intentionally status stratified.

Model results are saved to:

```text
reports/model_results.csv
```

## Model Interpretation

Feature importance outputs are saved under `reports/`:

```text
reports/logistic_regression_feature_importance.csv
reports/random_forest_feature_importance.csv
reports/xgboost_feature_importance.csv
```

Main interpretation findings:

- Enrollment related features appear consistently important across Logistic Regression, Random Forest, and XGBoost.
- Important enrollment features include `enrollment_type_ESTIMATED`, `enrollment_type_ACTUAL`, `enrollment_count`, and `log_enrollment_count`.
- Text length features, phase missingness, start year, and location or country features also contribute to model predictions.
- The dominance of enrollment related features is useful but requires caution because enrollment fields may be updated during or after a trial.

A future stricter model should test performance without enrollment related features to check whether the current model depends too heavily on potentially leaky fields.

## Dashboard Plan

The planned Streamlit dashboard will include:

| Tab | Purpose |
|---|---|
| Overview | Summary of trial counts, statuses, and risk distribution |
| Risk Explorer | Filter and explore trials by risk, sponsor, phase, geography, and study type |
| Model Insights | Show model metrics and feature importance |
| Data Quality | Display missingness, field coverage, and data quality warnings |
| Trial Detail | Inspect individual trial records and model inputs |

## How to Run

Install dependencies:

```powershell
uv sync
```

Pull sample data:

```powershell
uv run python src/clinical_trial_risk/ingest/pull_sample.py
```

Normalize raw data:

```powershell
uv run python src/clinical_trial_risk/clean/process_sample.py
```

Build the processed modeling dataset:

```powershell
uv run python src/clinical_trial_risk/features/build_features.py
```

Train baseline models and save model results:

```powershell
uv run python src/clinical_trial_risk/models/baseline.py
```

## Current Progress

Completed:

- Initial ClinicalTrials.gov sample pull
- Raw JSON output
- Nested JSON normalization
- Data dictionary notes
- Initial EDA
- Processed modeling dataset
- Baseline model comparison
- Feature importance outputs
- Model interpretation notes

In progress or planned:

- Larger and more realistic data pull
- Stricter leakage checks
- Feature importance stability testing
- Streamlit dashboard
- Optional PyTorch text model for eligibility criteria and brief summaries

## Limitations

This project uses public clinical trial metadata, which may be incomplete, inconsistently reported, or affected by reporting delays.

Important limitations:

- The current sample contains only 600 records.
- The current sample is intentionally status stratified.
- The target label is a simplified proxy for trial risk.
- The model does not directly predict recruitment difficulty yet.
- Some fields may be updated after trial launch, which creates potential leakage concerns.
- Text fields are currently represented through simple length and keyword features rather than deeper NLP methods.
- Model performance has not yet been validated on a larger or more realistic dataset.

Model outputs should be interpreted as exploratory risk signals, not as clinical, regulatory, or operational recommendations.

## Next Steps

Planned next steps:

1. Pull a larger dataset from ClinicalTrials.gov
2. Re run ingestion, normalization, feature engineering, and modeling
3. Test a stricter feature set without enrollment related fields
4. Compare feature importance stability across datasets
5. Build the Streamlit dashboard
6. Add trial level model explanations
7. Consider a PyTorch text model for trial summaries and eligibility criteria