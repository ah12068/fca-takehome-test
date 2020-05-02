# -*- coding: utf-8 -*-

import logging
import pandas as pd
from cleaner_functions import *
from constants import columns


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    raw_data = pd.read_csv(f'../../data/raw/loans_raw.csv')
    initial_row, initial_col = raw_data.shape
    logger.info(f'Cleaning data of size: {initial_row},{initial_col}')

    logger.info(f"Dropping cols w/ >=50% NaN's")
    drop_na_cols(raw_data, 0.5)

    logger.info(f'Cleaning text based data')
    lower_case_cols(raw_data)
    remove_whitespace(raw_data)

    logger.info(f'Cleaning data types')
    make_col_numeric(raw_data, 'credit_score')
    convert_datetime(raw_data, 'earliest_credit_line')
    raw_data.annual_income = raw_data.annual_income.apply(income_to_numeric)

    logger.info(f'Re-Categorising categorical variables')
    raw_data.employment_length = raw_data.employment_length.apply(categorise_employment_length)
    raw_data.home_ownership = raw_data.home_ownership.apply(categorise_home_ownership)
    raw_data.inquiries_6m = raw_data.inquiries_6m.apply(categorise_inquiries)
    raw_data.purpose = raw_data.purpose.apply(categorise_purpose)
    raw_data.term = raw_data.term.apply(categorise_term)

    logger.info(f'Imputing variables')
    impute_col(raw_data, 'total_current_balance', 'district', 'median')

    logger.info(f'Creating variables')
    create_credit_age(raw_data, 'earliest_credit_line', 2015)
    raw_data['class'] = raw_data.loan_status.apply(make_binary_class)

    logger.info(f'Dropping samples')
    drop_cols(raw_data, columns['drop'])
    drop_na_rows(raw_data)

    new_row, new_col = raw_data.shape

    logger.info(f'Clean data size: ({new_row},{new_col}). '
                f'Difference: row: {initial_row - new_row}, col: {initial_col - new_col}')

    output_filepath = f'../../data/interim/loans_clean.csv'
    raw_data.to_csv(output_filepath)
    logger.info(f'Clean data exported to {output_filepath}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
