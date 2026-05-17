from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    PROJECT_ROOT
    / 'data'
    / 'processed'
    / 'trials_modeling_larger_sample.csv'
)

MODEL_RESULTS_PATH = PROJECT_ROOT / 'reports' / 'model_results_larger.csv'

NO_ENROLLMENT_RESULTS_PATH = (
    PROJECT_ROOT
    / 'reports'
    / 'model_results_larger_no_enrollment.csv'
)

XGBOOST_FEATURE_IMPORTANCE_PATH = (
    PROJECT_ROOT
    / 'reports'
    / 'xgboost_feature_importance_larger.csv'
)

NO_ENROLLMENT_XGBOOST_FEATURE_IMPORTANCE_PATH = (
    PROJECT_ROOT
    / 'reports'
    / 'xgboost_feature_importance_larger_no_enrollment.csv'
)

st.set_page_config(
    page_title='Clinical Trial Risk Analytics',
    layout='wide',
)


@st.cache_data
def load_trials_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_report_data(path):
    if not path.exists():
        return None

    return pd.read_csv(path)


def count_values(df, column):
    return (
        df[column]
        .fillna('MISSING')
        .value_counts()
        .reset_index()
        .rename(columns={column: 'value', 'count': 'count'})
    )


def main():
    st.title('Clinical Trial Risk Analytics Platform')

    st.caption(
        'An interactive dashboard for exploring ClinicalTrials.gov trial metadata, '
        'risk labels, model results, feature importance, and data quality.'
    )

    trials_df = load_trials_data()

    overview_tab, risk_explorer_tab, model_insights_tab, data_quality_tab, trial_detail_tab = st.tabs(
        [
            'Overview',
            'Risk Explorer',
            'Model Insights',
            'Data Quality',
            'Trial Detail',
        ]
    )

    with overview_tab:
        st.header('Overview')

        total_trials = len(trials_df)
        high_risk_count = (trials_df['risk_label'] == 1).sum()
        low_risk_count = (trials_df['risk_label'] == 0).sum()
        unique_statuses = trials_df['status'].nunique()

        metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)

        metric_col_1.metric('Total Trials', f'{total_trials:,}')
        metric_col_2.metric('High Risk Trials', f'{high_risk_count:,}')
        metric_col_3.metric('Low Risk Trials', f'{low_risk_count:,}')
        metric_col_4.metric('Statuses', unique_statuses)

        st.subheader('Status Distribution')

        status_counts = count_values(trials_df, 'status')

        status_fig = px.bar(
            status_counts,
            x='value',
            y='count',
            text='count',
            labels={
                'value': 'Status',
                'count': 'Number of trials',
            },
            title='Trial Status Distribution',
        )

        st.plotly_chart(status_fig, width='stretch')

        st.subheader('Study Type Distribution')

        study_type_counts = count_values(trials_df, 'study_type')

        study_type_fig = px.bar(
            study_type_counts,
            x='value',
            y='count',
            text='count',
            labels={
                'value': 'Study type',
                'count': 'Number of trials',
            },
            title='Study Type Distribution',
        )

        st.plotly_chart(study_type_fig, width='stretch')

        st.subheader('Sample Rows')

        display_columns = [
            'nct_id',
            'title',
            'status',
            'risk_group',
            'phase',
            'study_type',
            'sponsor_class',
            'start_year',
        ]

        st.dataframe(
            trials_df[display_columns].head(25),
            width='stretch',
        )

    with risk_explorer_tab:
        st.header('Risk Explorer')

        st.write(
            'Filter the larger clinical trial sample by risk group, status, phase, '
            'study type, sponsor class, and sex eligibility.'
        )

        filter_col_1, filter_col_2, filter_col_3 = st.columns(3)

        with filter_col_1:
            selected_risk_groups = st.multiselect(
                'Risk group',
                options=sorted(trials_df['risk_group'].dropna().unique()),
                default=sorted(trials_df['risk_group'].dropna().unique()),
            )

            selected_statuses = st.multiselect(
                'Status',
                options=sorted(trials_df['status'].dropna().unique()),
                default=sorted(trials_df['status'].dropna().unique()),
            )

        with filter_col_2:
            selected_phases = st.multiselect(
                'Phase',
                options=sorted(trials_df['phase'].fillna('MISSING').unique()),
                default=sorted(trials_df['phase'].fillna('MISSING').unique()),
            )

            selected_study_types = st.multiselect(
                'Study type',
                options=sorted(trials_df['study_type'].fillna('MISSING').unique()),
                default=sorted(trials_df['study_type'].fillna('MISSING').unique()),
            )

        with filter_col_3:
            selected_sponsor_classes = st.multiselect(
                'Sponsor class',
                options=sorted(trials_df['sponsor_class'].fillna('MISSING').unique()),
                default=sorted(trials_df['sponsor_class'].fillna('MISSING').unique()),
            )

            selected_sexes = st.multiselect(
                'Sex eligibility',
                options=sorted(trials_df['sex'].fillna('MISSING').unique()),
                default=sorted(trials_df['sex'].fillna('MISSING').unique()),
            )

        filtered_df = trials_df[
            trials_df['risk_group'].isin(selected_risk_groups)
            & trials_df['status'].isin(selected_statuses)
            & trials_df['phase'].fillna('MISSING').isin(selected_phases)
            & trials_df['study_type'].fillna('MISSING').isin(selected_study_types)
            & trials_df['sponsor_class'].fillna('MISSING').isin(selected_sponsor_classes)
            & trials_df['sex'].fillna('MISSING').isin(selected_sexes)
        ]

        st.subheader('Filtered Results')

        st.metric('Matching Trials', f'{len(filtered_df):,}')

        if filtered_df.empty:
            st.warning('No trials match the selected filters.')
        else:
            risk_counts = count_values(filtered_df, 'risk_group')

            risk_fig = px.bar(
                risk_counts,
                x='value',
                y='count',
                text='count',
                labels={
                    'value': 'Risk group',
                    'count': 'Number of trials',
                },
                title='Risk Group Distribution for Filtered Trials',
            )

            st.plotly_chart(risk_fig, width='stretch')

            explorer_columns = [
                'nct_id',
                'title',
                'status',
                'risk_group',
                'phase',
                'study_type',
                'sponsor_class',
                'sex',
                'start_year',
                'num_locations',
                'num_countries',
            ]

            st.dataframe(
                filtered_df[explorer_columns],
                width='stretch',
                hide_index=True,
            )

    with model_insights_tab:
        st.header('Model Insights')

        st.write(
            'This section compares baseline model performance on the larger sample '
            'and shows how performance changes after removing enrollment-related fields.'
        )

        full_model_results_df = load_report_data(MODEL_RESULTS_PATH)
        no_enrollment_results_df = load_report_data(NO_ENROLLMENT_RESULTS_PATH)

        if full_model_results_df is None:
            st.warning(f'Could not find full model results at: {MODEL_RESULTS_PATH}')
        else:
            st.subheader('Larger Sample Model Results')
            st.dataframe(
                full_model_results_df.round(3),
                width='stretch',
                hide_index=True,
            )

            roc_auc_fig = px.bar(
                full_model_results_df,
                x='model',
                y='roc_auc',
                text='roc_auc',
                labels={
                    'model': 'Model',
                    'roc_auc': 'ROC-AUC',
                },
                title='ROC-AUC by Model, Larger Sample',
            )

            st.plotly_chart(roc_auc_fig, width='stretch')

        if no_enrollment_results_df is None:
            st.warning(
                f'Could not find no-enrollment model results at: {NO_ENROLLMENT_RESULTS_PATH}'
            )
        else:
            st.subheader('No-Enrollment Model Results')
            st.dataframe(
                no_enrollment_results_df.round(3),
                width='stretch',
                hide_index=True,
            )

            no_enrollment_roc_auc_fig = px.bar(
                no_enrollment_results_df,
                x='model',
                y='roc_auc',
                text='roc_auc',
                labels={
                    'model': 'Model',
                    'roc_auc': 'ROC-AUC',
                },
                title='ROC-AUC by Model, No-Enrollment Feature Set',
            )

            st.plotly_chart(no_enrollment_roc_auc_fig, width='stretch')

        st.subheader('Full Model vs No-Enrollment Model')

        if full_model_results_df is not None and no_enrollment_results_df is not None:
            full_comparison_df = full_model_results_df.copy()
            full_comparison_df['feature_set'] = 'With enrollment features'

            no_enrollment_comparison_df = no_enrollment_results_df.copy()
            no_enrollment_comparison_df['feature_set'] = 'No enrollment features'

            combined_results_df = pd.concat(
                [full_comparison_df, no_enrollment_comparison_df],
                ignore_index=True,
            )

            comparison_fig = px.bar(
                combined_results_df,
                x='model',
                y='roc_auc',
                color='feature_set',
                barmode='group',
                text='roc_auc',
                labels={
                    'model': 'Model',
                    'roc_auc': 'ROC-AUC',
                    'feature_set': 'Feature set',
                },
                title='ROC-AUC Comparison With and Without Enrollment Features',
            )

            st.plotly_chart(comparison_fig, width='stretch')

            st.info(
                'Removing enrollment-related fields caused model performance to drop. '
                'This suggests that enrollment fields carry strong predictive signal, '
                'but they may also introduce leakage if they are updated during or after a trial.'
            )

        st.subheader('XGBoost Feature Importance')

        xgboost_importance_df = load_report_data(XGBOOST_FEATURE_IMPORTANCE_PATH)

        if xgboost_importance_df is None:
            st.warning(
                f'Could not find XGBoost feature importance at: {XGBOOST_FEATURE_IMPORTANCE_PATH}'
            )
        else:
            top_xgboost_features = xgboost_importance_df.head(15)

            xgboost_importance_fig = px.bar(
                top_xgboost_features.sort_values('importance'),
                x='importance',
                y='feature',
                orientation='h',
                labels={
                    'importance': 'Importance',
                    'feature': 'Feature',
                },
                title='Top XGBoost Feature Importances, Larger Sample',
            )

            st.plotly_chart(xgboost_importance_fig, width='stretch')

        st.subheader('No-Enrollment XGBoost Feature Importance')

        no_enrollment_xgboost_importance_df = load_report_data(
            NO_ENROLLMENT_XGBOOST_FEATURE_IMPORTANCE_PATH
        )

        if no_enrollment_xgboost_importance_df is None:
            st.warning(
                'Could not find no-enrollment XGBoost feature importance at: '
                f'{NO_ENROLLMENT_XGBOOST_FEATURE_IMPORTANCE_PATH}'
            )
        else:
            top_no_enrollment_features = no_enrollment_xgboost_importance_df.head(15)

            no_enrollment_importance_fig = px.bar(
                top_no_enrollment_features.sort_values('importance'),
                x='importance',
                y='feature',
                orientation='h',
                labels={
                    'importance': 'Importance',
                    'feature': 'Feature',
                },
                title='Top XGBoost Feature Importances, No-Enrollment Feature Set',
            )

            st.plotly_chart(no_enrollment_importance_fig, width='stretch')

    with data_quality_tab:
        st.header('Data Quality')

        st.write(
            'This section summarizes missingness and data quality issues in the larger '
            'processed dataset. These checks help explain which fields are reliable, '
            'which fields need careful handling, and which fields may introduce leakage.'
        )

        st.subheader('Missing Values by Column')

        missing_summary = (
            trials_df.isna()
            .sum()
            .reset_index()
        )

        missing_summary.columns = ['column', 'missing_count']
        missing_summary['missing_percent'] = (
            missing_summary['missing_count'] / len(trials_df) * 100
        )

        missing_summary = missing_summary.sort_values(
            'missing_percent',
            ascending=False,
        )

        st.dataframe(
            missing_summary,
            width='stretch',
            hide_index=True,
        )

        top_missing = missing_summary[missing_summary['missing_count'] > 0].head(15)

        if top_missing.empty:
            st.success('No missing values found in the processed dataset.')
        else:
            missing_fig = px.bar(
                top_missing.sort_values('missing_percent'),
                x='missing_percent',
                y='column',
                orientation='h',
                text='missing_percent',
                labels={
                    'missing_percent': 'Missing percent',
                    'column': 'Column',
                },
                title='Top Missing Fields',
            )

            st.plotly_chart(missing_fig, width='stretch')

        st.subheader('Status Distribution')

        status_counts = count_values(trials_df, 'status')

        status_fig = px.bar(
            status_counts,
            x='value',
            y='count',
            text='count',
            labels={
                'value': 'Status',
                'count': 'Number of trials',
            },
            title='Status Distribution in Larger Sample',
        )

        st.plotly_chart(status_fig, width='stretch')

        st.subheader('Missingness Indicators')

        missing_indicator_columns = [
            'phase_missing',
            'maximum_age_missing',
            'locations_missing',
            'interventions_missing',
        ]

        available_missing_indicators = [
            column
            for column in missing_indicator_columns
            if column in trials_df.columns
        ]

        if available_missing_indicators:
            missing_indicator_summary = []

            for column in available_missing_indicators:
                missing_indicator_summary.append(
                    {
                        'indicator': column,
                        'count': int(trials_df[column].sum()),
                        'percent': float(trials_df[column].mean() * 100),
                    }
                )

            missing_indicator_df = pd.DataFrame(missing_indicator_summary)

            st.dataframe(
                missing_indicator_df,
                width='stretch',
                hide_index=True,
            )

            indicator_fig = px.bar(
                missing_indicator_df.sort_values('percent'),
                x='percent',
                y='indicator',
                orientation='h',
                text='percent',
                labels={
                    'percent': 'Percent of trials',
                    'indicator': 'Missingness indicator',
                },
                title='Missingness Indicator Rates',
            )

            st.plotly_chart(indicator_fig, width='stretch')
        else:
            st.info('No missingness indicator columns found in the processed dataset.')

        st.subheader('Data Quality Notes')

        st.markdown(
            '''
            - The larger sample is still status-stratified, so it should not be interpreted as the real-world distribution of clinical trial outcomes.
            - Missing values may carry meaning. For example, missing phase may indicate that phase is not applicable rather than simply missing data.
            - Enrollment-related fields were strong predictors in the broader metadata model, but they may be updated during or after a trial.
            - The no-enrollment model is a more conservative check for pre-outcome prediction, but its performance is lower.
            - Model outputs should be treated as exploratory risk signals rather than clinical or operational recommendations.
            '''
        )

    with trial_detail_tab:
        st.header('Trial Detail')

        st.write(
            'Select an individual trial to inspect its metadata, risk label, '
            'and engineered features.'
        )

        selected_nct_id = st.selectbox(
            'Select NCT ID',
            options=sorted(trials_df['nct_id'].dropna().unique()),
        )

        selected_trial = trials_df[trials_df['nct_id'] == selected_nct_id].iloc[0]

        st.subheader(selected_trial['title'])

        metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)

        metric_col_1.metric('Status', selected_trial['status'])
        metric_col_2.metric('Risk Group', selected_trial['risk_group'])
        metric_col_3.metric('Study Type', selected_trial['study_type'])
        metric_col_4.metric('Start Year', int(selected_trial['start_year']) if pd.notna(selected_trial['start_year']) else 'Missing')

        st.subheader('Core Trial Metadata')

        metadata_col_1, metadata_col_2 = st.columns(2)

        with metadata_col_1:
            st.write('**NCT ID**')
            st.write(selected_trial['nct_id'])

            st.write('**Phase**')
            st.write(selected_trial['phase'])

            st.write('**Sponsor Class**')
            st.write(selected_trial['sponsor_class'])

            st.write('**Sex Eligibility**')
            st.write(selected_trial['sex'])

        with metadata_col_2:
            st.write('**Number of Conditions**')
            st.write(selected_trial['num_conditions'])

            st.write('**Number of Interventions**')
            st.write(selected_trial['num_interventions'])

            st.write('**Number of Locations**')
            st.write(selected_trial['num_locations'])

            st.write('**Number of Countries**')
            st.write(selected_trial['num_countries'])

        st.subheader('Engineered Features')

        engineered_features = {
            'Eligibility Text Length': selected_trial['eligibility_text_length'],
            'Brief Summary Length': selected_trial['brief_summary_length'],
            'Has Placebo': selected_trial['has_placebo'],
            'Adult Only': selected_trial['adult_only'],
            'Phase Missing': selected_trial['phase_missing'],
            'Maximum Age Missing': selected_trial['maximum_age_missing'],
            'Locations Missing': selected_trial['locations_missing'],
            'Interventions Missing': selected_trial['interventions_missing'],
        }

        engineered_features_df = pd.DataFrame(
            engineered_features.items(),
            columns=['Feature', 'Value'],
        )

        st.dataframe(
            engineered_features_df,
            width='stretch',
            hide_index=True,
        )

        st.subheader('Raw Trial Row')

        raw_trial_df = (
            selected_trial
            .astype(str)
            .reset_index()
        )

        raw_trial_df.columns = ['field', 'value']

        st.dataframe(
            raw_trial_df,
            width='stretch',
            hide_index=True,
        )


if __name__ == '__main__':
    main()