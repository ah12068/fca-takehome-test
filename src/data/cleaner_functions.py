from pandas import to_datetime
from numpy import nan


def lower_case_cols(df):
    cols_to_lower = df.select_dtypes(include='object').columns
    for col in cols_to_lower:
        df[col] = df[col].str.lower()

    return


def remove_whitespace(df):
    cols_to_clean = df.select_dtypes(include='object').columns
    for col in cols_to_clean:
        df[col] = df[col].str.strip()

    return


def make_col_numeric(df, col):
    df[col] = df[col].astype(int)

    return


def income_to_numeric(annual_income):
    if len(annual_income.split(' ')) == 1:
        return float(annual_income)
    if len(annual_income.split(' ')) > 1:
        return float(annual_income.split(' ')[-1])
    else:
        return np.nan


def convert_datetime(df, col):
    df[col] = to_datetime(df[col])

    return


def categorise_employment_length(employment_length):
    prefix = 'EL'

    if employment_length in ['10+ years']:
        return f'{prefix}_10plus'
    if employment_length in ['< 1 year']:
        return f'{prefix}_less1'

    if employment_length in ['1 year', '2 years', '3 years']:
        return f'{prefix}_1to3'
    if employment_length in ['4 years', '5 years', '6 years']:
        return f'{prefix}_4to6'
    if employment_length in ['7 years', '8 years', '9 years']:
        return f'{prefix}_7to9'


def categorise_home_ownership(home_ownership):
    if home_ownership in ['mortgage', 'rent', 'own']:
        return home_ownership
    else:
        return f'other'


def categorise_inquiries(inquiries_6m):
    if inquiries_6m == 0:
        return f'no inquiry'
    if inquiries_6m == 1:
        return f'1 inquiry'
    else:
        return f'2+ inquiry'


def categorise_purpose(purpose):
    if purpose not in ['debt_consolidation', 'credit_card']:
        return f'other'
    else:
        return purpose


def categorise_term(term):
    return float(term.split(' ')[0])


def make_binary_class(loan_status):
    if loan_status == 'fully paid':
        return 0
    if loan_status == 'ongoing':
        return nan

    if loan_status == 'charged off':
        return 1
    if loan_status == 'late (> 90 days)':
        return 1
    if loan_status == 'default':
        return 1


def create_credit_age(df, col, cutoff):
    df['credit_age'] = cutoff - df[col].dt.year

    return


def impute_col(df, col, segment, method):
    if col and segment not in df.columns:
        raise ValueError(f'columns not in dataframe, was given {col} {segment}')

    if method == 'median':
        fill_method = method
        fill_value = df[col].median()
    elif method == 'mean':
        fill_method = method
        fill_value = df[col].mean()
    else:
        raise ValueError(f'method takes only "median" or "mean", was given {method}')

    filler = df.groupby(f'{segment}')[col].transform(fill_method)
    df[col].fillna(filler, inplace=True)

    if df[col].isna().any() == True:
        df[col].fillna(fill_value, inplace=True)
    else:
        pass

    return


def drop_na_cols(df, pct_thresh):
    row, col = df.shape
    threshold = row * pct_thresh

    df.dropna(axis=1, thresh=threshold, inplace=True)

    return


def drop_na_rows(df):
    df.dropna(axis=0, inplace=True)

    return


def drop_cols(df, cols_to_drop):
    df.drop(labels=cols_to_drop, axis=1, inplace=True)
    return
