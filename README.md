# Clinical Trial Risk Analytics Platform

An end to end data science project that uses public ClinicalTrials.gov records to analyze clinical trial outcome risk, train baseline prediction models, investigate possible leakage, and display results in an interactive Streamlit dashboard.

## Overview

Clinical trials can terminate early, withdraw before enrollment, or become suspended for many reasons, including recruitment difficulty, study complexity, eligibility restrictions, sponsor constraints, geography, and trial design.

This project builds a reproducible clinical trial analytics pipeline that moves from raw public API data to a modeling ready dataset, baseline machine learning models, feature importance outputs, leakage aware model comparisons, and an interactive dashboard.

The goal is not to provide clinical, regulatory, or operational recommendations. The goal is to build a transparent data science workflow for exploring trial risk signals from public trial metadata.

## Data Source

Primary data source:

- ClinicalTrials.gov public study records
- Data is pulled through the ClinicalTrials.gov API
- Raw API output is saved as JSON
- Nested records are normalized into flat tables for analysis and modeling

## Target Variable

The first version uses trial status as a simplified proxy for risk.

| Trial Status | Risk Label | Meaning |
|---|---:|---|
| COMPLETED | 0 | Low risk |
| TERMINATED | 1 | High risk |
| WITHDRAWN | 1 | High risk |
| SUSPENDED | 1 | High risk |

Other statuses, such as recruiting, not yet recruiting, active not recruiting, and unknown are excluded or held out for future modeling decisions.

This target is intentionally simple. It does not capture every form of clinical trial difficulty, such as slow recruitment, delayed completion, poor reporting, or inconclusive outcomes.

## Project Workflow

1. Pull clinical trial records from ClinicalTrials.gov
2. Save raw API responses as JSON
3. Normalize nested trial records into flat tables
4. Perform exploratory data analysis
5. Build a processed modeling dataset
6. Train baseline classification models
7. Save model results and feature importance outputs
8. Run a larger sample validation
9. Run a no enrollment leakage check
10. Build a Streamlit dashboard for exploration

## Project Structure

| Path | Purpose |
|---|---|
| data/raw/ | Raw ClinicalTrials.gov API outputs |
| data/interim/ | Normalized flat trial tables |
| data/processed/ | Modeling ready datasets |
| notebooks/ | Data dictionary, EDA notes, modeling notes, and interpretation notes |
| src/clinical_trial_risk/ingest/ | API data pulling |
| src/clinical_trial_risk/clean/ | Raw JSON normalization |
| src/clinical_trial_risk/features/ | Feature engineering |
| src/clinical_trial_risk/models/ | Baseline modeling and interpretation |
| src/clinical_trial_risk/dashboard/ | Streamlit dashboard |
| reports/ | Model result CSVs and feature importance CSVs |

## Feature Engineering

The processed modeling dataset includes structured, text derived, and missingness based features.

| Feature | Description |
|---|---|
| enrollment_count | Reported trial enrollment count |
| log_enrollment_count | Log transformed enrollment count |
| num_conditions | Number of listed conditions |
| num_interventions | Number of listed interventions |
| num_locations | Number of listed study locations |
| num_countries | Number of unique countries |
| eligibility_text_length | Character length of eligibility criteria |
| brief_summary_length | Character length of brief summary |
| start_year | Parsed trial start year |
| has_placebo | Keyword indicator for placebo |
| adult_only | Indicator for adult or older adult studies |
| phase_missing | Indicator for missing phase |
| maximum_age_missing | Indicator for missing maximum age |
| locations_missing | Indicator for missing location text |
| interventions_missing | Indicator for missing intervention text |

Categorical features include:

- phase
- study_type
- sponsor_class
- sex
- enrollment_type

Completion related fields such as completion_date, primary_completion_date, and trial_duration_days are excluded from the first modeling pass because they may leak information about the final trial outcome.

## Exploratory Data Analysis

The initial EDA focused on:

- Dataset structure
- Target distribution
- Missingness
- Missingness by status
- Categorical feature distributions
- Numeric feature distributions
- Text length features
- Date fields
- Feature readiness
- Leakage risks

Main EDA takeaways:

- The initial sample contained 600 clinical trial records.
- The larger sample contains 2,000 status stratified trial records.
- The samples are intentionally status stratified, so they should not be interpreted as the real world distribution of clinical trial outcomes.
- Missingness is manageable, but some missing values may carry meaning.
- phase and maximum_age require explicit missing value handling.
- Completion based fields may introduce target leakage and should be handled carefully.

## Modeling

The first modeling pipeline compares:

1. Dummy Classifier
2. Logistic Regression
3. Random Forest
4. XGBoost

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1
- ROC AUC
- Confusion matrix

Accuracy alone is not sufficient because the target is imbalanced.

## Larger Sample Results

The larger sample uses 500 records per target status, for 2,000 total records.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Dummy Classifier | 0.750 | 0.750 | 1.000 | 0.857 | 0.500 |
| Logistic Regression | 0.840 | 0.968 | 0.813 | 0.884 | 0.910 |
| Random Forest | 0.835 | 0.947 | 0.827 | 0.883 | 0.906 |
| XGBoost | 0.852 | 0.906 | 0.897 | 0.901 | 0.916 |

Main findings:

- All real models outperform the Dummy Classifier on ROC AUC.
- XGBoost performs best overall on the larger sample, with 0.916 ROC AUC and 0.901 F1.
- Logistic Regression remains competitive and interpretable.
- Model performance remains strong after scaling from the initial 600 record sample to a 2,000 record sample.

These results are preliminary because the larger sample is still status stratified.

## No Enrollment Leakage Check

Feature importance showed that enrollment related fields were among the strongest predictors.

Removed features:

- enrollment_count
- log_enrollment_count
- enrollment_type

The goal was to test whether models still performed well after removing fields that may be updated during or after a trial.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Dummy Classifier | 0.750 | 0.750 | 1.000 | 0.857 | 0.500 |
| Logistic Regression | 0.620 | 0.866 | 0.583 | 0.697 | 0.680 |
| Random Forest | 0.640 | 0.793 | 0.703 | 0.746 | 0.653 |
| XGBoost | 0.728 | 0.751 | 0.953 | 0.840 | 0.653 |

Main findings:

- Removing enrollment related fields caused performance to drop substantially.
- This suggests that enrollment fields carry a large amount of predictive signal.
- The drop supports the leakage concern because enrollment fields may be updated during or after a trial.
- The no enrollment models still beat the dummy baseline on ROC AUC, but performance is much weaker.
- The project now has two modeling views:
  - A broader reported metadata model with stronger performance
  - A stricter no enrollment model with lower performance but reduced leakage risk

## Model Interpretation

Feature importance outputs are saved in reports/.

The broader metadata model found that enrollment related fields were consistently important across models.

Important features included:

- enrollment_type_ESTIMATED
- enrollment_type_ACTUAL
- enrollment_count
- log_enrollment_count
- eligibility_text_length
- brief_summary_length
- start_year
- phase_missing
- num_locations
- num_countries

After enrollment features were removed, the models shifted toward:

- Text length features
- Start year
- Phase and phase missingness
- Sponsor class
- Location and country counts
- Placebo indicator
- Sex eligibility

This makes the project more credible because it does not only report the highest model score. It also investigates whether the strongest predictors are safe to use for a realistic pre outcome prediction task.

## Dashboard

The project includes a Streamlit dashboard for exploring:

- Dataset overview
- Risk labels and status distribution
- Trial filters
- Model results
- Full model vs no enrollment model comparison
- Feature importance
- Data quality issues
- Individual trial details

Dashboard tabs:

| Tab | Purpose |
|---|---|
| Overview | Shows high level dataset metrics and distributions |
| Risk Explorer | Filters trials by status, risk group, phase, study type, sponsor class, and sex |
| Model Insights | Compares model results and feature importance |
| Data Quality | Shows missingness and data quality notes |
| Trial Detail | Lets users inspect one trial at a time |

Run the dashboard with:

    uv run streamlit run src/clinical_trial_risk/dashboard/app.py

## How to Run

Install dependencies:

    uv sync

Pull the larger sample:

    uv run python src/clinical_trial_risk/ingest/pull_larger_sample.py

Normalize the larger raw data:

    uv run python src/clinical_trial_risk/clean/process_larger_sample.py

Build the larger processed modeling dataset:

    uv run python src/clinical_trial_risk/features/build_larger_features.py

Train baseline models with enrollment features:

    uv run python src/clinical_trial_risk/models/baseline_larger.py

Train no enrollment models:

    uv run python src/clinical_trial_risk/models/baseline_larger_no_enrollment.py

Run the dashboard:

    uv run streamlit run src/clinical_trial_risk/dashboard/app.py

## Current Progress

Completed:

- ClinicalTrials.gov API data pull
- Pagination for larger data pulls
- Raw JSON output
- Nested JSON normalization
- Data dictionary notes
- Initial EDA
- Processed modeling dataset
- Baseline model comparison
- Larger sample validation
- Feature importance outputs
- No enrollment leakage check
- Streamlit dashboard

Planned or optional next steps:

- Add screenshots to the README
- Deploy the Streamlit dashboard
- Test a less artificially balanced sample
- Add trial level predicted risk scores
- Add a PyTorch text model for eligibility criteria and brief summaries
- Improve dashboard styling and layout

## Limitations

This project uses public clinical trial metadata, which may be incomplete, inconsistently reported, or affected by reporting delays.

Important limitations:

- The current datasets are status stratified and do not represent real world clinical trial outcome rates.
- The target label is a simplified proxy for trial risk.
- The model does not directly predict recruitment difficulty yet.
- Some fields may be updated after trial launch, creating potential leakage concerns.
- Enrollment related fields are highly predictive but may not be safe for strict pre outcome prediction.
- Text fields are currently represented through simple length and keyword features rather than deeper NLP methods.
- Model performance has not yet been validated on a naturally distributed sample.

Model outputs should be interpreted as exploratory risk signals, not clinical, regulatory, or operational recommendations.
