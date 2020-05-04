from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTENC, ADASYN, RandomOverSampler
import numpy as np
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
    "LogisticRegression": LogisticRegression(random_state=random_seed, n_jobs=cpu_count),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=random_seed),
    "RandomForestClassifier": RandomForestClassifier(random_state=random_seed, n_jobs=cpu_count)
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
]

parameter_grid = {
    "LogisticRegression": {
        "penalty": ['l2'],
        "C": [0.001, 0.01, 0.1, 1, 10, 100]
    },
    "DecisionTreeClassifier": {
        "max_leaf_nodes": list(range(2, 100)),
        "min_samples_split": [5, 10, 100, 500],
    },
    "RandomForestClassifier": {
        'n_estimators': [100, 300, 500, 1000],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 20, 50, 100],
        'min_samples_split': [10, 100, 500],
        'min_samples_leaf': [1, 10, 100]
    }
}

model_metrics = {
    'AUC':'roc_auc',
    'RECALL':'recall',
    'PRECISION':'precision',
    'F1':'f1'
}