import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploratory Data Analysis

    This document explores the normalized ClinicalTrials.gov sample. The goal is to understand dataset structure, label distribution, missingness, feature reliability, and early modeling risks before building the first processed modeling dataset.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt

    return mo, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset Snapshot

    The normalized sample contains 600 clinical trial records and 26 columns. Each row represents one clinical trial study record. The core columns include trial ID, title, status, phase, study type, enrollment information, sponsor information, conditions, interventions, locations, eligibility criteria, and brief summary.
    """)
    return


@app.cell
def _(pd):
    df_raw = pd.read_csv('data/interim/trials_normalized_sample.csv')

    df = df_raw.copy()

    status_to_risk_label = {
        'COMPLETED': 0,
        'TERMINATED': 1,
        'WITHDRAWN': 1,
        'SUSPENDED': 1,
    }

    risk_label_to_group = {
        0: 'low risk',
        1: 'high risk',
    }

    df['risk_label'] = df['status'].map(status_to_risk_label)
    df['risk_group'] = df['risk_label'].map(risk_label_to_group)

    df['eligibility_text_length'] = df['eligibility_text'].fillna('').str.len()
    df['brief_summary_length'] = df['brief_summary'].fillna('').str.len()

    df['start_date_parsed'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['completion_date_parsed'] = pd.to_datetime(df['completion_date'], errors='coerce')
    df['start_year'] = df['start_date_parsed'].dt.year
    df['trial_duration_days'] = (
        df['completion_date_parsed'] - df['start_date_parsed']
    ).dt.days

    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The normalized input dataset contains 600 rows and 26 original columns. After adding temporary EDA helper columns such as `risk_label`, `risk_group`, text lengths, parsed dates, and trial duration, the working EDA dataframe contains more columns. Each `nct_id` appears to be unique, so the table appears to represent one row per clinical trial. The dataset contains four selected statuses: completed, terminated, withdrawn, and suspended.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Sanity Checks

    Before feature engineering, I checked whether the normalized table has the expected structure. The main goal is to confirm that each row represents one clinical trial and that the trial identifier is unique.
    """)
    return


@app.cell
def _(df):
    sanity_checks = {
        'rows': len(df),
        'columns': df.shape[1],
        'unique_nct_ids': df['nct_id'].nunique(),
        'duplicate_nct_ids': df['nct_id'].duplicated().sum(),
        'unique_statuses': df['status'].nunique(),
    }

    sanity_checks
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sampling Note

    This sample was intentionally pulled with equal counts for completed, terminated, withdrawn, and suspended trials. Because of that, the status distribution is useful for early model development, but it should not be interpreted as the real-world distribution of ClinicalTrials.gov outcomes.
    """)
    return


@app.cell(hide_code=True)
def _(df):
    status_counts = df['status'].value_counts()

    ax = status_counts.plot(kind='bar', title='Trial Status Distribution')
    return (ax,)


@app.cell(hide_code=True)
def _(ax):
    ax.set_xlabel('Status')
    ax.set_ylabel('Number of trials')
    ax.bar_label(ax.containers[0])
    return


@app.cell(hide_code=True)
def _(df):
    risk_counts = df["risk_group"].value_counts()

    ax1 = risk_counts.plot(kind="bar", title="Grouped Risk Label Distribution")
    return (ax1,)


@app.cell(hide_code=True)
def _(ax1):
    ax1.set_xlabel("Risk group")
    ax1.set_ylabel("Number of trials")
    ax1.bar_label(ax1.containers[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The grouped target is intentionally imbalanced because three statuses are mapped to high risk and only completed trials are mapped to low risk. This is acceptable for the first modeling version, but evaluation should use precision, recall, F1, and ROC-AUC rather than accuracy alone.
    """)
    return


@app.cell
def _(df):
    df.isnull().sum()
    return


@app.cell
def _(df, pd):
    nulls_summary = pd.DataFrame({
        'null_count' : df.isnull().sum(), 
        'null_proportion' : round(df.isnull().mean(), 3)
    
    }).sort_values('null_proportion', ascending = False)

    nulls_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - `phase` missingness is high, but that can be expected since not all studies have phases (example: observational)
    - `maximum_age` missingness is high but not all studies have a maximum age requirement. For feature engineering, missing `maximum_age` may need to be handled separately because it could represent "no maximum age specified."
    - `locations`/`countries` missingness is the same, suggesting that rows that are missing locations are also missing countries. Geographic features are usable still but nulls must be handled
    - `interventions` is missing for about 9.2% of rows. This may be expected for some observational studies or records where intervention information is not reported in the same way as interventional trials. This field should be checked against `study_type` before deciding how to handle it.
    - `completion_date` and `primary_completion_date` have relatively low missingness, around 5%. These fields are mostly available, but they should be used carefully because they may introduce target leakage if the goal is to predict trial risk before the final trial outcome is known.
    - `minimum_age`, `start_date`, `enrollment_type`, and `enrollment_count` have low missingness. These fields appear reliable enough for early feature engineering, though date parsing and enrollment type interpretation still need to be checked.
    """)
    return


@app.cell
def _(df):
    missing_by_status = (
        df[['status', 'phase', 'maximum_age', 'locations', 'countries', 'interventions',
           'completion_date', 'primary_completion_date', 'minimum_age',
           'start_date', 'enrollment_type', 'enrollment_count']].groupby('status')
        .apply(lambda group: group.isna().mean().round(3))
        .T
    )

    missing_by_status
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Missingness by status helps identify whether some fields are systematically less complete for high risk trials. If missingness differs strongly by status, missingness itself may become predictive, but it may also introduce bias.
    """)
    return


@app.cell(hide_code=True)
def _(df, plt):
    df.groupby('study_type').count()['nct_id'].plot.bar()
    plt.title('Bar chart of study_type')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Most trials in this sample are interventional
    - Interventional and observational trials may have different missingness patterns, design complexities, and risk profiles.
    - `study_type` should be included as an early categorical feature.
    """)
    return


@app.cell(hide_code=True)
def _(df, plt):
    df.groupby('phase').count()['nct_id'].plot.barh()
    plt.title('Bar chart of phase')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - `phase` has high missingness, so it should not be treated as a fully complete feature.
    - Among trials with a phase listed, Phase 2, Phase 1, and Phase 3 appear commonly represented.
    - Some records contain combined phases, such as `PHASE1; PHASE2` or `PHASE2; PHASE3`, which may need to be handled during feature engineering.
    - For the first modeling dataset, missing phase values should likely be filled with `MISSING` or `NOT_APPLICABLE` rather than dropping those rows.
    """)
    return


@app.cell(hide_code=True)
def _(df, plt):
    df.groupby('sponsor_class').count()['nct_id'].plot.barh()
    plt.title('Bar chart of sponsor_class')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Sample is dominated by sponsor class OTHER with INDUSTRY as second largest group.
    - Appears usable as a categorical feature but rare classes may need to be grouped or handled in some other way during modeling.
    """)
    return


@app.cell(hide_code=True)
def _(df, plt):
    df.groupby('sex').count()['nct_id'].plot.barh()
    plt.title('Bar chart of sex')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Most trials are open to all sexes.
    - Female/male only trials are much less common, so while it could be useful, some categories might be sparse.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Risk Rate by Categorical Features

    After checking basic categorical distributions, I compared high risk rate across several categorical columns. This helps identify whether trial design, sponsor class, phase, or sex eligibility may be associated with trial outcome.

    Because this sample was intentionally stratified by status, these patterns should be treated as exploratory rather than representative of real-world risk rates.
    """)
    return


@app.cell
def _(df):
    def risk_rate_by_category(column):
        temp = df[[column, 'risk_label', 'nct_id']].copy()
        temp[column] = temp[column].fillna('MISSING')

        return (
            temp.groupby(column)
            .agg(
                trial_count=('nct_id', 'count'),
                high_risk_rate=('risk_label', 'mean'),
            )
            .sort_values('trial_count', ascending=False)
        )

    risk_rate_by_category('phase')
    return (risk_rate_by_category,)


@app.cell
def _(risk_rate_by_category):
    risk_rate_by_category('study_type')
    return


@app.cell
def _(risk_rate_by_category):
    risk_rate_by_category('sponsor_class')
    return


@app.cell
def _(risk_rate_by_category):
    risk_rate_by_category('sex')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These categorical risk-rate tables are useful for feature planning, but they should not be overinterpreted. Some categories have small sample sizes, and the sample was intentionally pulled by status. For modeling, categorical fields such as `study_type`, `phase`, `sponsor_class`, and `sex` appear usable, but rare categories may need to be grouped or handled carefully.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Numeric Feature Exploration

    I inspected numeric and count-based fields to understand their ranges, missingness, and potential usefulness for modeling.
    """)
    return


@app.cell
def _(df):
    numeric_columns = [
        'enrollment_count',
        'num_conditions',
        'num_interventions',
        'num_locations',
        'num_countries',
    ]

    df[numeric_columns].describe()
    return


@app.cell
def _(df):
    ax2 = df['enrollment_count'].plot(
        kind='hist',
        bins=30,
        title='Enrollment Count Distribution'
    )
    ax2.set_xlabel('Enrollment count')
    return


@app.cell
def _(df):
    ax3 = df['num_locations'].plot(
        kind='hist',
        bins=30,
        title='Number of Locations Distribution'
    )
    ax3.set_xlabel('Number of locations')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `enrollment_count` is likely to be skewed because clinical trials can range from very small studies to very large studies. This means a log-transformed enrollment feature may be useful later. Count-based features such as `num_conditions`, `num_interventions`, `num_locations`, and `num_countries` appear useful as simple proxies for trial complexity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Text Field Exploration

    I inspected the two main text fields, `eligibility_text` and `brief_summary`, by creating simple text length features. The goal is not to do NLP yet, but to check whether these fields are usable for simple structured features and later text modeling.
    """)
    return


@app.cell
def _(df):
    df['eligibility_text_length'] = df['eligibility_text'].fillna('').str.len()
    df['brief_summary_length'] = df['brief_summary'].fillna('').str.len()

    df[['eligibility_text_length', 'brief_summary_length']].describe()
    return


@app.cell(hide_code=True)
def _(df):
    ax4 = df['eligibility_text_length'].plot(
        kind='hist',
        bins=30,
        title='Eligibility Text Length Distribution'
    )
    ax4.set_xlabel('Eligibility text length')
    return


@app.cell(hide_code=True)
def _(df):
    ax5 = df['brief_summary_length'].plot(
        kind='hist',
        bins=30,
        title='Brief Summary Length Distribution'
    )
    ax5.set_xlabel('Brief summary length')
    return


@app.cell
def _(df):
    df[["eligibility_text_length", "brief_summary_length"]].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Eligibility criteria and brief summaries appear useful for later text-based feature engineering. For the first structured model, simple text length features can act as rough proxies for study complexity. Later versions can use these text fields for NLP-based modeling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Date Field Sanity Check

    I inspected parsed start dates, completion dates, start year, and trial duration. These fields are useful, but completion-based fields need special caution because they may not be available at prediction time.
    """)
    return


@app.cell
def _(df, pd):
    df['start_date_parsed'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['completion_date_parsed'] = pd.to_datetime(df['completion_date'], errors='coerce')

    df['start_year'] = df['start_date_parsed'].dt.year
    df['trial_duration_days'] = (
        df['completion_date_parsed'] - df['start_date_parsed']
    ).dt.days

    df[['start_year', 'trial_duration_days']].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The negative duration check helps identify impossible or suspicious date records where the completion date appears before the start date. These records would need to be reviewed or excluded before using `trial_duration_days` as a feature.
    """)
    return


@app.cell
def _(df):
    negative_duration_count = (df['trial_duration_days'] < 0).sum()
    negative_duration_count
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this sample, the negative duration count is 0, so no obviously invalid trial durations were found.
    """)
    return


@app.cell(hide_code=True)
def _(df):
    ax6 = df['start_year'].value_counts().sort_index().plot(
        kind='barh',
        title='Trials by Start Year'
    )
    ax6.set_xlabel("Number of trials")
    ax6.set_ylabel("Start year")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Date fields are useful but need careful handling. `start_year` appears useful as a basic temporal feature. However, `completion_date` and `trial_duration_days` may introduce target leakage if the goal is to predict risk before the final trial outcome is known. For the first pre-outcome model, completion-related fields should either be excluded or clearly separated from features available at trial start.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature Readiness Summary

    | Feature | Initial Assessment | Planned Handling |
    |---|---|---|
    | `study_type` | Usable | One-hot encode or categorical encode |
    | `sponsor_class` | Usable with rare categories | Group rare categories if needed |
    | `phase` | Useful but high missingness | Fill missing as `MISSING` or `NOT_APPLICABLE` |
    | `enrollment_count` | Useful but likely skewed | Consider log transform |
    | `num_conditions` | Usable | Use as trial complexity feature |
    | `num_interventions` | Usable | Use as trial complexity feature |
    | `num_locations` | Usable with missing handling | Add missing indicator if needed |
    | `num_countries` | Usable with missing handling | Add missing indicator if needed |
    | `minimum_age` | Mostly usable | Parse later if needed |
    | `maximum_age` | High missingness | Treat missing carefully, possibly as no upper age specified |
    | `eligibility_text_length` | Useful | Create from eligibility text |
    | `brief_summary_length` | Useful | Create from brief summary |
    | `start_year` | Useful | Extract from start date |
    | `completion_date` | Leakage risk | Avoid in first pre-outcome model |
    | `trial_duration_days` | Leakage risk | Avoid or separate from pre-outcome model |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EDA Conclusions

    - The normalized dataset is structurally usable for a first modeling pass.
    - The sample contains 600 unique clinical trial records and 26 original columns.
    - The sample was intentionally stratified by status, so it should not be interpreted as the real-world distribution of trial outcomes.
    - The binary target maps completed trials to low risk and terminated, withdrawn, and suspended trials to high risk.
    - Missingness is manageable, but some missing values likely carry meaning rather than simply representing bad data.
    - `phase` and `maximum_age` require explicit missing value handling.
    - `study_type`, `sponsor_class`, enrollment information, condition counts, intervention counts, and location counts appear useful for initial feature engineering.
    - Text fields such as `eligibility_text` and `brief_summary` are useful candidates for simple text length features and later NLP modeling.
    - Completion-related fields may introduce target leakage and should be handled carefully.
    - The next step is to build a processed modeling dataset with `risk_label` and simple engineered features.
    """)
    return


if __name__ == "__main__":
    app.run()
