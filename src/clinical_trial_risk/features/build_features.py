from pathlib import Path

import numpy as np
import pandas as pd 

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = PROJECT_ROOT / 'data' / 'interim' / 'trials_normalized_sample.csv'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_PATH = PROCESSED_DATA_DIR / 'trials_modeling_sample.csv'

STATUS_TO_RISK_LABEL = {
    'COMPLETED': 0, 
    'TERMINATED' : 1, 
    'WITHDRAWN' : 1, 
    'SUSPENDED' : 1, 
}

RISK_LABEL_TO_GROUP = {
    0 : 'low risk', 
    1 : 'high risk'
}

def contains_keyword(series, keyword):
    return series.fillna('').str.contains(keyword, case=False, regex=False).astype(int)
    
def build_features(df):
    features_df = df.copy()
    features_df['risk_label'] = features_df['status'].map(STATUS_TO_RISK_LABEL)
    features_df['risk_group'] = features_df['risk_label'].map(RISK_LABEL_TO_GROUP)

    categorical_cols = [
        'phase', 
        'study_type', 
        'sponsor_class', 
        'sex', 
        'enrollment_type', 
    ]

    for col in categorical_cols:
        features_df[col] = features_df[col].fillna('MISSING')

    features_df['enrollment_count'] = pd.to_numeric(features_df['enrollment_count'], errors='coerce')

    features_df['log_enrollment_count'] = np.log1p(features_df['enrollment_count'].clip(lower=0))

    features_df['eligibility_text_length'] = features_df['eligibility_text'].fillna('').str.len()

    features_df['brief_summary_length'] = features_df['brief_summary'].fillna('').str.len()

    features_df['start_date_parsed'] = pd.to_datetime(features_df['start_date'], errors='coerce', format='mixed')

    features_df['start_year'] = features_df['start_date_parsed'].dt.year

    combined_text = features_df['title'].fillna('') + ' ' + features_df['interventions'].fillna('') + ' ' + features_df['brief_summary'].fillna('') + ' ' + features_df['eligibility_text'].fillna('')

    features_df['has_placebo'] = contains_keyword(combined_text, 'placebo')

    std_ages = features_df['std_ages'].fillna('')

    features_df['adult_only'] = ((std_ages.str.contains('ADULT', case=False, regex=False) | std_ages.str.contains('OLDER_ADULT', case=False, regex=False)) & ~std_ages.str.contains('CHILD', case=False, regex=False)).astype(int)

    features_df['phase_missing'] = (features_df['phase'] == 'MISSING').astype(int)
    features_df['maximum_age_missing'] = (features_df['maximum_age'] == 'MISSING').astype(int)
    features_df['locations_missing'] = (features_df['locations'] == 'MISSING').astype(int)
    features_df['interventions_missing'] = (features_df['interventions'] == 'MISSING').astype(int)

    output_columns = [
        'nct_id',
        'title',
        'status',
        'risk_label',
        'risk_group',
        'phase',
        'study_type',
        'sponsor_class',
        'sex',
        'enrollment_type',
        'enrollment_count',
        'log_enrollment_count',
        'num_conditions',
        'num_interventions',
        'num_locations',
        'num_countries',
        'eligibility_text_length',
        'brief_summary_length',
        'start_year',
        'has_placebo',
        'adult_only',
        'phase_missing',
        'maximum_age_missing',
        'locations_missing',
        'interventions_missing',
    ]

    return features_df[output_columns]

def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    features_df = build_features(df)

    features_df.to_csv(OUTPUT_PATH, index=False)

    print(f'Loaded normalized data from: {INPUT_PATH}')
    print(f'Input shake: {df.shape}')
    print(f'Output shape: {features_df.shape}')
    print(f'Saved processed modeling dataset to: {OUTPUT_PATH}')

    print('\nRisk label distribution:')
    print(features_df['risk_label'].value_counts(dropna=False))

    print('\nMissing values in processed dataset:')
    print(features_df.isna().sum().sort_values(ascending=False))

if __name__ == '__main__':
    main()