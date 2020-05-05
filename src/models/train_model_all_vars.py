import logging
import pandas as pd
import pickle as pkl
from constants import (
    baseline_classifiers,
    feature_coefs,
    sampling_strats
)
from functions import baseline_trainer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


strat = 'ADASYN'


def main(sampling_strat):

    logger = logging.getLogger(__name__)

    if sampling_strat not in sampling_strats.keys():
        raise ValueError(f'Invalid sampling strategy, only {[x for x in sampling_strats.keys()]}')

    df = pd.read_csv('../../data/processed/loans_processed.csv')
    sampler = sampling_strats[f'{sampling_strat}']

    logger.info(f'Training baseline models')

    for classifier in baseline_classifiers.keys():
        logger.info(f'Classifier: {classifier} w/ {sampling_strat}')
        if classifier in feature_coefs:
            algorithm, metrics, visual_report = baseline_trainer(
                processed_df=df,
                algorithm=baseline_classifiers[classifier],
                sampler=sampler,
                cf='features',
                name=classifier
        )


        else:
            algorithm, metrics, visual_report = baseline_trainer(
                processed_df=df,
                algorithm=baseline_classifiers[classifier],
                sampler=sampler,
                cf='coefficients',
                name=classifier
        )

        f = open(f'../../models/{classifier}_{sampling_strat}_allvar_metrics.pickle', 'wb')
        pkl.dump(metrics, f)
        f.close()

        m_out = open(f'../../models/{classifier}_{sampling_strat}_allvar_model.pickle', 'wb')
        pkl.dump(algorithm, m_out)
        m_out.close()

        visual_report.write_html(f'../../reports/figures/{classifier}_{sampling_strat}_allvar_report.html')
        logger.info(f'Plotly charts exported to reports/figures')

        logger.info(f'{classifier} and metrics (pickle files) exported to models/')

    return algorithm


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main(sampling_strat=strat)


