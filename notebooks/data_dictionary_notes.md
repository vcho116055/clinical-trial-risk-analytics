# Data Dictionary Notes

These notes describe the first normalized clinical trial table created from the raw ClinicalTrials.gov API sample.

Output file:

```text
data/interim/trials_normalized_sample.csv
```

Each row represents one clinical trial study record.

## Purpose

This file is meant to help explain what each column in the normalized dataset means.

The current dataset is an intermediate dataset. It is cleaner than the raw JSON from the API, but it is not the final modeling dataset yet.

## Columns

### nct_id

The ClinicalTrials.gov identifier for the study.

This is the unique trial ID assigned by ClinicalTrials.gov. It usually starts with `NCT`, followed by numbers.

Example:

```text
NCT00000000
```

This field is useful as the primary identifier for looking up, joining, or referencing a trial.

---

### title

The brief title of the clinical trial.

This is a short, public facing title that describes the study. It usually includes the condition, intervention, population, or purpose of the trial.

---

### status

The overall recruitment status of the trial.

For this project, the first target variable is based on this field.

Important statuses for the first version:

```text
COMPLETED
TERMINATED
WITHDRAWN
SUSPENDED
```

Definitions:

```text
COMPLETED
The study concluded normally. Participants are no longer receiving the intervention or being examined.

TERMINATED
The study was stopped early and will not resume. Participants are no longer receiving the intervention or being examined.

WITHDRAWN
The study was stopped early before the first participant was enrolled.

SUSPENDED
The study was stopped early but may resume in the future.
```

Initial project labeling:

```text
Low risk = COMPLETED

High risk = TERMINATED, WITHDRAWN, SUSPENDED
```

This is a simplified starting label. It does not capture every kind of trial risk, and it should be treated as a practical first version target rather than a perfect definition of clinical trial failure.

---

### phase

The clinical trial phase.

This usually describes the stage of testing for a drug, biological product, or intervention.

Examples may include:

```text
EARLY_PHASE1
PHASE1
PHASE2
PHASE3
PHASE4
```

Some studies may not have a phase, especially observational studies or studies where phase does not apply.

This field may be useful because trial risk can vary by phase.

---

### study_type

The type of clinical study.

Common values include:

```text
INTERVENTIONAL
OBSERVATIONAL
EXPANDED_ACCESS
```

Interventional studies assign participants to interventions according to a protocol.

Observational studies observe outcomes in predefined groups without assigning the intervention.

This field matters because interventional and observational studies may have different risk patterns.

---

### enrollment_count

The number of participants enrolled or expected to be enrolled in the trial.

This may be an actual count or an estimated count depending on the study record.

This field may be useful because larger studies can have different operational risks than smaller studies.

---

### enrollment_type

Whether the enrollment count is actual or estimated.

Possible examples:

```text
ACTUAL
ESTIMATED
```

This helps interpret `enrollment_count`.

---

### start_date

The date the study started or was expected to start.

This can be used later to create features such as:

```text
start_year
trial_duration_days
```

---

### completion_date

The date the study was completed or was expected to be completed.

This is not necessarily the date results were published. It usually refers to the study completion date in the clinical trial record.

This can be used with `start_date` to estimate trial duration.

---

### primary_completion_date

The date when final data collection was completed for the primary outcome measure.

This may be different from `completion_date`.

This field may be useful later because primary completion can happen before full study completion.

---

### sponsor_name

The name of the lead sponsor responsible for the clinical trial.

Examples may include a pharmaceutical company, university, hospital, government agency, or research organization.

This field may be useful for grouping trials by sponsor.

---

### sponsor_class

The type or category of the lead sponsor.

Examples may include:

```text
INDUSTRY
NIH
FED
OTHER_GOV
OTHER
```

This field may be useful because trial risk patterns may differ between industry sponsored, academic, government, and other sponsor types.

---

### conditions

The diseases, conditions, or health related issues being studied.

Examples:

```text
Breast Cancer
Diabetes Mellitus
COVID-19
Hypertension
Healthy Volunteers
```

This field describes what the trial is about medically.

It may be useful later for condition level analysis or broad condition category features.

---

### num_conditions

The number of conditions listed for the trial.

This is a simple engineered count based on the `conditions` field.

A trial studying multiple conditions may be more complex than a trial studying one condition.

---

### interventions

The treatments, drugs, devices, procedures, behavioral programs, diagnostic tests, or other interventions being studied.

Examples:

```text
Drug A
Placebo
Behavioral Counseling
Surgery
Diagnostic Test
Medical Device
```

This field describes what is being tested or studied in the trial.

For observational studies, this field may be empty or less central.

---

### num_interventions

The number of interventions listed for the trial.

This is a simple engineered count based on the interventions list.

A higher number of interventions may suggest a more complex trial design.

---

### locations

The study site locations included in the trial record.

In this project, this field is currently normalized from available facility, city, and country information.

This field can help identify whether a trial is single site, multi site, domestic, or international.

---

### num_locations

The number of listed study locations.

This may be useful because multi site trials can be operationally different from single site trials.

---

### countries

The countries represented across the listed trial locations.

This field is useful for identifying whether a trial spans multiple countries.

---

### num_countries

The number of unique countries represented in the trial locations.

This may be useful as a proxy for geographic complexity.

---

### sex

The sex eligibility requirement for the trial.

Possible examples:

```text
ALL
FEMALE
MALE
```

This field describes who is eligible to participate based on sex.

---

### minimum_age

The minimum eligible age for participants.

Example:

```text
18 Years
```

This can be used later to create features such as adult only trials.

---

### maximum_age

The maximum eligible age for participants.

Example:

```text
65 Years
```

This field may be missing if there is no upper age limit.

---

### std_ages

Standardized age categories for eligible participants.

Examples may include:

```text
CHILD
ADULT
OLDER_ADULT
```

This field can help create simpler age eligibility features.

---

### eligibility_text

The full eligibility criteria text for the trial.

This usually includes inclusion criteria and exclusion criteria.

This field describes who can and cannot participate in the study.

Examples of information that may appear here:

```text
Required diagnosis
Age requirements
Prior treatment history
Health conditions
Pregnancy restrictions
Medication restrictions
Lab value requirements
```

This is a useful text field for later feature engineering because longer or more restrictive eligibility criteria may be related to recruitment difficulty.

Possible future features:

```text
eligibility_text_length
has_exclusion_criteria
adult_only
requires_prior_treatment
```

---

### brief_summary

A short public summary of the clinical trial.

This usually describes the purpose of the study, the condition being studied, and sometimes the intervention or hypothesis.

This field can be used for text based features later.

Possible future features:

```text
brief_summary_length
text embeddings
condition or intervention keywords
```

## Current Target Variable Plan

The current target variable is based on trial status.

```text
Low risk = COMPLETED

High risk = TERMINATED, WITHDRAWN, SUSPENDED
```

This is intentionally simple for the first version.

The goal is not to perfectly define clinical trial risk yet. The goal is to create a usable supervised learning target so the project can move from data ingestion into cleaning, feature engineering, modeling, and evaluation.

## Planned Feature Engineering Ideas

Possible features to build later:

```text
trial_duration_days
enrollment_count
num_conditions
num_interventions
num_locations
num_countries
phase
study_type
sponsor_type
eligibility_text_length
brief_summary_length
start_year
has_placebo
adult_only
```

## Notes and Limitations

This first data dictionary is intentionally rough.

The purpose is to understand the fields in the first normalized sample, not to finalize the full modeling dataset.

Some fields may be missing for some trials because ClinicalTrials.gov records vary by study type, trial phase, sponsor, and reporting completeness.

The current target variable is a simplified proxy for clinical trial risk. It treats completed trials as lower risk and terminated, withdrawn, or suspended trials as higher risk.

This is useful for the first modeling version, but it does not capture all forms of clinical trial difficulty, such as slow recruitment, delayed completion, poor reporting, inconclusive outcomes, or trial completion with major operational issues.