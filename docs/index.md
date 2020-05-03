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

## Features

- `account_id` - `int64` - Unique identifier for accounts
- `class` - `float64` - Binary class for those who `1` Fully paid back their debt or `0` Defaulted, Late (>90 days)) and Charged-off
- `purpose_credit_card` - `uint8` - Binary who has purpose `credit_card`
- `purpose_debt_consolidation` - `uint8` - Binary who has purpose `debt_condolidation`
- `purpose_other` - `uint8` - Binary who has purpose `other`
- `home_ownership_mortgage` - `uint8` - Binary who has home ownership of `mortgage` 
- `home_ownership_other` - `uint8` - Binary who has home ownership of `other`
- `home_ownership_own` - `uint8` - Binary who has home ownership of `own`
- `home_ownership_rent` - `uint8` - Binary who has home ownership of `rent`
- `employment_length_EL_10plus` - `uint8` - Binary who has employment length of `10+ years` 
- `employment_length_EL_1to3` - `uint8` - Binary who has employment length of `1-3 years`
- `employment_length_EL_4to6` - `uint8` - Binary who has employment length of `4-6 years`
- `employment_length_EL_7to9` - `uint8` - Binary who has employment length of `7-9 years`
- `employment_length_EL_less1` - `uint8` - Binary who has employment length of `<1 year`
- `inquiries_6m_1 inquiry` - `uint8` - Binary who made `1` inquiries
- `inquiries_6m_2+ inquiry` - `uint8` - Binary who made `2+` inquiries
- `inquiries_6m_no inquiry` - `uint8` - Binary who made `0` inquiries
- `installment` - `float64` - See original data dictionary
- `loan_amount` - `float64` - See original data dictionary
- `interest_rate` - `float64` - See original data dictionary
- `term` - `float64` - Binary, `0` if term is `36 months` else `1` for `60 months`
- `annual_income` - `float64` - See original data dictionary
- `public_records` - `float64` - See original data dictionary
- `delinquency_2y` - `float64` - See original data dictionary
- `open_accounts` - `float64` - See original data dictionary
- `debt_to_income` - `float64` - See original data dictionary
- `credit_card_usage` - `float64` - See original data dictionary
- `credit_card_balance` - `float64` - See original data dictionary
- `total_current_balance` - `float64` - See original data dictionary, some values are imputed via the median and segmented by district 
- `nr_accounts` - `float64` - See original data dictionary
- `credit_score` - `float64` - See original data dictionary
- `credit_age` - `float64` - Difference between `earliest_credit_line` and `2015` latest year in dataset.


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

## Variables that were dropped for Machine Learning 

- `description`: >50% null
- `last_record_months`: >50% null
- `last_delinquency_months`: >50% null
- `last_derog_months`: >50% null
- `title`: text field, non-NLP task 
- `job_title`: text field, non-NLP task
- `district`: text field, non-NLP task
- `postcode_district`: text field, non-NLP task
- `loan_status`: Data leakage, when predicting new application this would not be known
- `year`: Data leakage, when predicting new application this would not be known 
- `amount payed`: Data leakage, when predicting new application this would not be known
- `earliest_credit_line`: Date field, used to create `credit_age`

