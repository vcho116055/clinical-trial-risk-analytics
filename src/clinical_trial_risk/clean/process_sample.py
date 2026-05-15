import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_INPUT_PATH = PROJECT_ROOT / 'data' / 'raw' / 'clinical_trials_sample.json'
INTERIM_DATA_DIR = PROJECT_ROOT / 'data' / 'interim'
OUTPUT_PATH = INTERIM_DATA_DIR / 'trials_normalized_sample.csv'

def get_nested_value(dictionary, keys, default=None):
    current_val = dictionary

    for key in keys:
        if not isinstance(current_val, dict):
            return default
        
        current_val = current_val.get(key)

        if current_val is None:
            return default
        
    return current_val

def join_list_values(vals):
    if not vals:
        return ''
    
    if not isinstance(vals, list):
        return str(vals)
    
    return '; '.join(str(val) for val in vals if val is not None)

def extract_location_summary(locs):
    if not locs:
        return {
            'locations' : '', 
            'num_locations' : 0, 
            'countries' : '', 
            'num_countries' : 0, 
        }
    
    loc_names = []
    countries = set()

    for loc in locs: 
        facility = get_nested_value(loc, ['facility'], '')
        city = get_nested_value(loc, ['city'], '')
        country = get_nested_value(loc, ['country'], '')

        loc_parts = [part for part in [facility, city, country] if part]
        loc_name = ', '.join(loc_parts)

        if loc_name:
            loc_names.append(loc_name)
        
        if country:
            countries.add(country)

    return {
        'locations' : '; '.join(loc_names), 
        'num_locations' : len(locs), 
        'countries' : '; '.join(sorted(countries)), 
        'num_countries' : len(countries)
    }

def extract_intervention_names(interventions):
    if not interventions:
        return ''
    
    names = []

    for inter in interventions:
        name = inter.get('name')

        if name:
            names.append(name)

    return '; '.join(names)

def normalize_study(study):
    protocol = study.get('protocolSection', {})

    identification = protocol.get('identificationModule', {})
    status = protocol.get('statusModule', {})
    design = protocol.get('designModule', {})
    sponsor = protocol.get('sponsorCollaboratorsModule', {})
    conditions = protocol.get('conditionsModule', {})
    interventions = protocol.get('armsInterventionsModule', {})
    contacts_locations = protocol.get('contactsLocationsModule', {})
    eligibility = protocol.get('eligibilityModule', {})
    description = protocol.get('descriptionModule', {})

    condition_list = conditions.get('conditions', [])
    intervention_list = interventions.get('interventions', [])
    location_list = contacts_locations.get('locations', [])

    location_summary = extract_location_summary(location_list)

    return {
        'nct_id': identification.get('nctId'),
        'title': identification.get('briefTitle'),
        'status': status.get('overallStatus'),
        'phase': join_list_values(design.get('phases')),
        'study_type': design.get('studyType'),
        'enrollment_count': get_nested_value(design, ['enrollmentInfo', 'count']),
        'enrollment_type': get_nested_value(design, ['enrollmentInfo', 'type']),
        'start_date': get_nested_value(status, ['startDateStruct', 'date']),
        'completion_date': get_nested_value(status, ['completionDateStruct', 'date']),
        'primary_completion_date': get_nested_value(status, ['primaryCompletionDateStruct', 'date']),
        'sponsor_name': get_nested_value(sponsor, ['leadSponsor', 'name']),
        'sponsor_class': get_nested_value(sponsor, ['leadSponsor', 'class']),
        'conditions': join_list_values(condition_list),
        'num_conditions': len(condition_list),
        'interventions': extract_intervention_names(intervention_list),
        'num_interventions': len(intervention_list),
        'locations': location_summary['locations'],
        'num_locations': location_summary['num_locations'],
        'countries': location_summary['countries'],
        'num_countries': location_summary['num_countries'],
        'sex': eligibility.get('sex'),
        'minimum_age': eligibility.get('minimumAge'),
        'maximum_age': eligibility.get('maximumAge'),
        'std_ages': join_list_values(eligibility.get('stdAges')),
        'eligibility_text': eligibility.get('eligibilityCriteria'),
        'brief_summary': description.get('briefSummary'),
    }

def main():
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with RAW_INPUT_PATH.open('r', encoding='utf-8') as file:
        raw_data = json.load(file)

    studies = raw_data.get('studies', [])

    normalized_records = []

    for study in studies:
        normalized_records.append(normalize_study(study))

    trials_df = pd.DataFrame(normalized_records)

    trials_df.to_csv(OUTPUT_PATH, index=False)

    print(f'Loaded {len(studies)} raw studies')
    print(f'Created dataframe with shape: {trials_df.shape}')
    print(f'Saved normalized table to: {OUTPUT_PATH}')

    print()
    print('Columns:')
    print(list(trials_df.columns))

    print()
    print('Status distribution:')
    print(trials_df['status'].value_counts(dropna=False))

if __name__ == '__main__':
    main()