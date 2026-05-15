from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


API_URL = 'https://clinicaltrials.gov/api/v2/studies'

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
OUTPUT_PATH = RAW_DATA_DIR / 'clinical_trials_sample.json'

TARGET_STATUSES = [
    'COMPLETED',
    'TERMINATED',
    'WITHDRAWN',
    'SUSPENDED',
]

PAGE_SIZE_PER_STATUS = 150


def fetch_trials_by_status(status: str, page_size: int = PAGE_SIZE_PER_STATUS) -> list[dict[str, Any]]:
    '''Gets a small sample of trials for one ClinicalTrials.gov status. Returns list of dictionaries of string-any type key-value pairs.'''
    params = {
        'format' : 'json',
        'pageSize' : page_size,
        'query.term' : f'AREA[OverallStatus]{status}',
    }

    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    return data.get('studies', [])

def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_studies: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}

    for status in TARGET_STATUSES: 
        print(f'Pulling {PAGE_SIZE_PER_STATUS} trials with status: {status}')

        studies = fetch_trials_by_status(status)
        all_studies.extend(studies)
        status_counts[status] = len(studies)

        print(f' Retrieved {len(studies)} studies')

    output = {
        'metadata': {
            'source' : 'ClinicalTrials.gov API v2', 
            'api_url' : API_URL, 
            'pulled_at_utc' : datetime.now(timezone.utc).isoformat(),
            'target_statuses' : TARGET_STATUSES, 
            'page_size_per_status' : PAGE_SIZE_PER_STATUS, 
            'status_counts' : status_counts, 
            'total_studies' : len(all_studies), 
        }, 
        'studies' : all_studies
    }

    with OUTPUT_PATH.open('w', encoding='utf-8') as file:
        json.dump(output, file, indent=2)
    
    print()
    print(f'Saved {len(all_studies)} total studies to:')
    print(OUTPUT_PATH)

if __name__ == '__main__': 
    main()