# FCA assessment Documentation

## Scripts in src/data

* `make_dataset.py` - Cleans dataset .
* `constants.py` - Contains static variables.
* `cleaner_functions.py` - Contains functions to clean dataset.


## Scripts in src/features

* `make_features` - Creates features from dataset suitable for machine learning.


## Scripts in src/models

* `train.py` - Trains a machine learning model on the dataset.
* `predict.py` - Predicts a sample from the trained machine learning model.


## Variables - Post cleansed

- `account_id`: See original data-dictionary
- `installment`: See original data-dictionary
- `loan_amount`: See original data-dictionary
- `interest_rate`: See original data-dictionary
- `term`: See original data-dictionary
- `purpose`: See original data-dictionary
- `home_ownership`: See original data-dictionary
- `annual_income`: See original data-dictionary
- `employment_length`: See original data-dictionary
- `public_records`: See original data-dictionary
- `delinquency_2y`: See original data-dictionary
- `inquiries_6m`: See original data-dictionary
- `open_accounts`: See original data-dictionary
- `debt_to_income`: See original data-dictionary
- `credit_card_usage`: See original data-dictionary
- `credit_card_balance`: See original data-dictionary
- `total_current_balance`: See original data-dictionary
- `nr_accounts`: See original data-dictionary
- `credit_score`: See original data-dictionary
- `credit_age`: Number of years from earliest_credit_line till 2015
- `class`: {0,1} 0: Customer paid back loan, 1: Customer defaulted loan payment

