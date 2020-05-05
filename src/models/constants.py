from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTENC, ADASYN, RandomOverSampler
import pandas as pd
import os
import plotly.io as pio

PLOTLY_TEMPLATE = pio.templates.default = 'plotly_white'
PANDAS_TEMPLATE = pd.set_option('display.float_format', '{:.5f}'.format)
cpu_count = os.cpu_count()
random_seed = 1

processed_df = pd.read_csv('../../data/processed/loans_processed.csv')
id_col = ['account_id']
target_col = ["class"]
classes = ["Not Default", "Default"]
cate_cols = processed_df.nunique()[processed_df.nunique() == 2].keys().tolist()
cate_cols = [col for col in cate_cols if col not in target_col]
cate_cols_idx = [processed_df.columns.get_loc(col) for col in cate_cols]

baseline_classifiers = {
    "LogisticRegression": SGDClassifier(
        loss='log',
        shuffle=True,
        random_state=random_seed,
        n_jobs=cpu_count,
        early_stopping=True,
        average=True),

    # "HuberRegression": SGDClassifier(
    #     loss='modified_huber',
    #     shuffle=True,
    #     random_state=random_seed,
    #     n_jobs=cpu_count,
    #     early_stopping=True,
    #     average=True),

    "DecisionTreeClassifier": DecisionTreeClassifier(
        random_state=random_seed),
}

sampling_strats = {
    'RandomOverSampler': RandomOverSampler(sampling_strategy='minority'),
    'SMOTENC': SMOTENC(sampling_strategy='minority',
                       random_state=random_seed,
                       n_jobs=cpu_count,
                       categorical_features=cate_cols_idx
                       ),
    'ADASYN': ADASYN(sampling_strategy='minority',
                     random_state=random_seed,
                     n_jobs=cpu_count
                     ),

}

feature_coefs = [
    "DecisionTreeClassifier",
    "RandomForestClassifier"
]

alphas = [0.0001, 0.001, 0.01, 0.1]

parameter_grid = {
    "LogisticRegression": {
        'alpha': alphas
    },
    "DecisionTreeClassifier": {
        "max_depth": list(range(10, 210, 10)),
        "min_samples_split": [10, 100, 500, 1000],
    },
    "HuberRegression": {
        'alpha': alphas,
    }
}

model_metrics = {
    'AUC':'roc_auc',
    'RECALL':'recall',
    'PRECISION':'precision',
    'F1':'f1'
}