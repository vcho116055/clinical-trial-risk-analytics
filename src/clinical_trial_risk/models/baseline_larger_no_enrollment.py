from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'trials_modeling_larger_sample.csv'

TARGET_COLUMN = 'risk_label'

REPORTS_DIR = PROJECT_ROOT / 'reports'

MODEL_RESULTS_PATH = REPORTS_DIR / 'model_results_larger_no_enrollment.csv'

LOGISTIC_FEATURE_IMPORTANCE_PATH = (
    REPORTS_DIR / 'logistic_regression_feature_importance_larger_no_enrollment.csv'
)

RANDOM_FOREST_FEATURE_IMPORTANCE_PATH = (
    REPORTS_DIR / 'random_forest_feature_importance_larger_no_enrollment.csv'
)

XGBOOST_FEATURE_IMPORTANCE_PATH = (
    REPORTS_DIR / 'xgboost_feature_importance_larger_no_enrollment.csv'
)

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    'phase',
    'study_type',
    'sponsor_class',
    'sex',
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_dataset():
    return pd.read_csv(INPUT_PATH)


def validate_columns(df):
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = []

    for column in required_columns:
        if column not in df.columns:
            missing_columns.append(column)

    if missing_columns:
        raise ValueError(f'Missing required columns: {missing_columns}')
    
def create_preprocessor():
    numeric_transformer = Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy = 'median')),
            ('scaler', StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy = 'constant', fill_value= 'MISSING')),
            ('one_hot_encoder', OneHotEncoder(handle_unknown = 'ignore')), 
        ]
    )

    preprocessor = ColumnTransformer(
        transformers = [
            ('numeric', numeric_transformer, NUMERIC_FEATURES), 
            ('categorical', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor

def evaluate_model(model_name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'model' : model_name, 
        'accuracy' : accuracy_score(y_test, y_pred), 
        'precision' : precision_score(y_test, y_pred, zero_division=0), 
        'recall' : recall_score(y_test, y_pred, zero_division=0), 
        'f1' : f1_score(y_test, y_pred, zero_division = 0), 
        'roc_auc' : roc_auc_score(y_test, y_prob), 
    }

    print()
    print(f'{model_name} metrics:')
    for metric_name, metric_value in metrics.items():
        if metric_name != 'model':
            print(f'{metric_name}: {metric_value:.3f}')
    print()
    print(f'{model_name} confusion matrix:')
    print(confusion_matrix(y_test, y_pred))

    return metrics

def save_logistic_feature_importance(model, output_path):
    preprocessor = model.named_steps['preprocessor']
    logistic_regression = model.named_steps['model']

    feature_names = preprocessor.get_feature_names_out()
    coefficients = logistic_regression.coef_[0]

    feature_importance_df = pd.DataFrame(
        {
            'feature' : feature_names, 
            'coefficient' : coefficients,
        }
    )

    feature_importance_df['abs_coefficient'] = feature_importance_df['coefficient'].abs()

    feature_importance_df['direction'] = feature_importance_df['coefficient'].apply(lambda value: 'higher_risk' if value > 0 else 'lower_risk')

    feature_importance_df = feature_importance_df.sort_values('abs_coefficient', ascending=False)

    feature_importance_df.to_csv(output_path, index=False)

    print()
    print(f'Saved Logistic Regression feature importance to: {output_path}')

    print()
    print('Top Logistic Regression feature importances:')
    print(feature_importance_df.head(15).round(3))

    return feature_importance_df

def save_tree_feature_importance(model_name, model, output_path):
    preprocessor = model.named_steps['preprocessor']
    tree_model = model.named_steps['model']

    feature_names = preprocessor.get_feature_names_out()
    importances = tree_model.feature_importances_

    feature_importance_df = pd.DataFrame(
        {
            'feature': feature_names,
            'importance': importances,
        }
    )

    feature_importance_df = feature_importance_df.sort_values(
        'importance',
        ascending=False,
    )

    feature_importance_df.to_csv(output_path, index=False)

    print()
    print(f'Saved {model_name} feature importance to: {output_path}')

    print()
    print(f'Top {model_name} feature importances:')
    print(feature_importance_df.head(15).round(3))

    return feature_importance_df

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    validate_columns(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f'Loaded dataset from: {INPUT_PATH}')
    print(f'Dataset shape: {df.shape}')
    print(f'Feature matrix shape: {X.shape}')
    print(f'Target shape: {y.shape}')

    print()
    print('Target distribution:')
    print(y.value_counts(dropna=False))

    print()
    print('Train/test split:')
    print(f'X_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'y_test shape: {y_test.shape}')

    print()
    print('Training target distribution:')
    print(y_train.value_counts(dropna=False))

    print()
    print('Test target distribution:')
    print(y_test.value_counts(dropna=False))

    print()
    print('Missing values in selected features:')
    print(X.isna().sum().sort_values(ascending=False))

    results = []

    dummy_model = Pipeline(
        steps=[
            ('preprocessor', create_preprocessor()),
            ('model', DummyClassifier(strategy='most_frequent')),
        ]
    )

    dummy_model.fit(X_train, y_train)

    results.append(
    evaluate_model(
        'Dummy Classifier',
        dummy_model,
        X_test,
        y_test,
    )
    )

    logistic_model = Pipeline(
        steps=[
            ('preprocessor', create_preprocessor()),
            (
                'model',
                LogisticRegression(
                    max_iter=1000,
                    class_weight='balanced',
                    random_state=42,
                ),
            ),
        ]
    )

    logistic_model.fit(X_train, y_train)

    results.append(
        evaluate_model(
            'Logistic Regression',
            logistic_model,
            X_test,
            y_test,
        )
    )

    save_logistic_feature_importance(
        logistic_model,
        LOGISTIC_FEATURE_IMPORTANCE_PATH,
    )

    random_forest_model = Pipeline(
        steps=[
            ('preprocessor', create_preprocessor()),
            (
                'model',
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_leaf=5,
                    random_state=42,
                    class_weight='balanced',
                ),
            ),
        ]
    )

    random_forest_model.fit(X_train, y_train)

    results.append(
        evaluate_model(
            'Random Forest',
            random_forest_model,
            X_test,
            y_test,
        )
    )

    save_tree_feature_importance(
        'Random Forest',
        random_forest_model,
        RANDOM_FOREST_FEATURE_IMPORTANCE_PATH,
    )

    xgboost_model = Pipeline(
        steps=[
            ('preprocessor', create_preprocessor()),
            (
                'model',
                XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric='logloss',
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    xgboost_model.fit(X_train, y_train)

    results.append(
        evaluate_model(
            'XGBoost',
            xgboost_model,
            X_test,
            y_test,
        )
    )

    save_tree_feature_importance(
        'XGBoost',
        xgboost_model,
        XGBOOST_FEATURE_IMPORTANCE_PATH,
    )

    results_df = pd.DataFrame(results)

    print()
    print('Model comparison:')
    print(results_df.round(3))

    results_df.to_csv(MODEL_RESULTS_PATH, index=False)

    print()
    print(f"Saved model results to: {MODEL_RESULTS_PATH}")


if __name__ == '__main__':
    main()