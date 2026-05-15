# Clinical Trial Risk and Enrollment Analytics Platform

## PRoblem 

Clinical trials can fail, terminate, or struggle due to a variety of issues such as recruitment problems, study complexity, sponsor constraints, geography, eligibility criteria, and trial design. This project builds an end to end data science platform that uses public clinical trial data to predict trial risk and visualize risk patterns through an interactive dashboard. 

## Data Source 

Primary data source preliminarily: ClinicalTrials.gov public study records. 

## Target Variable 

Initial target:

- High risk: terminated, withdrawn, suspended
- Low risk: completed

Recruiting, not yet recruiting, active not recruiting, unknown, and other statuses excluded/handled separately during modeling. 

## Planned Workflow 

1. Data ingestion 
2. Data cleaning and validation
3. Feature engineering
4. Exploratory data analysis
5. Baseline modeling 
6. Traditional ML models
7. TensorFlow and Pytorch experiments 
8. Model eval and interpretation
9. Interactive dashboard 
10. Limitations and next steps

## Planned Features

Trial phase, study type, enrollment count, sponsor type, number of conditions, number of interventions, number of locations, number of countries, trial duration, eligibility criteria length, summary length, age elegibility, sex elibility, and start year. 

## Models

Planned model comparison:

1. Dummy baseline
2. Logistic regression
3. Random forest
4. XGBoost
5. TensorFlow tabular neural network
6. PyTorch text model 

## Dashboard Plan

Planned Streamlit dashboard tabs: 

- Overview
- Risk Explorer
- Model Insights
- Trial Detail

## Limitations 

This project uses public trial metadata, which may be incomplete, inconsistently reported, or affected by reporting delays. Model outputs should be interpreted as exploratory risk signals rather than clinical or operational recommendations.
