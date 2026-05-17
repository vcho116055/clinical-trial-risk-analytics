# Day 3 Modeling Notes

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
- enrollment_count
- log_enrollment_count
- num_conditions
- num_interventions
- num_locations
- num_countries
- eligibility_text_length
- brief_summary_length
- start_year
- has_placebo
- adult_only
- phase_missing
- maximum_age_missing
- locations_missing
- interventions_missing