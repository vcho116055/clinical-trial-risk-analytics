from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


API_URL = 'https://clinicaltrials.gov/api/v2/studies'

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
OUTPUT_PATH = RAW_DATA_DIR / 'clinical_trials_larger_sample.json'

TARGET_STATUSES = [
    'COMPLETED',
    'TERMINATED',
    'WITHDRAWN',
    'SUSPENDED',
]

MAX_RECORDS_PER_STATUS = 500
PAGE_SIZE = 100


def fetch_trials_by_status(status, max_records, page_size = PAGE_SIZE):
    collected = []
    page_token = None

    while len(collected) < max_records:
        remaining = max_records - len(collected)
        current = min(page_size, remaining)

        params = {
            'format' : 'json',
            'pageSize' : current,
            'query.term' : f'AREA[OverallStatus]{status}',
        }

        if page_token:
            params['pageToken'] = page_token

        response = requests.get(API_URL, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        studies = data.get('studies', [])

        if not studies:
            break

        collected.extend(studies)

        page_token = data.get('nextPageToken')

        if not page_token:
            break

    return collected

def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_studies: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}

    for status in TARGET_STATUSES: 
        print(f'Pulling up to {MAX_RECORDS_PER_STATUS} trials with status: {status}')

        studies = fetch_trials_by_status(status, MAX_RECORDS_PER_STATUS, PAGE_SIZE)
        all_studies.extend(studies)
        status_counts[status] = len(studies)

        print(f' Retrieved {len(studies)} studies')

    output = {
        'metadata': {
            'source' : 'ClinicalTrials.gov API v2', 
            'api_url' : API_URL, 
            'pulled_at_utc' : datetime.now(timezone.utc).isoformat(),
            'target_statuses' : TARGET_STATUSES, 
            'max_records_per_status': MAX_RECORDS_PER_STATUS,
            'page_size': PAGE_SIZE, 
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