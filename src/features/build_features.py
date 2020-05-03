import logging
import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder,
    RobustScaler
)


def main():
    logger = logging.getLogger(__name__)

    output_file_path = f'../../data/processed/'

    logger.info(f'Reading data')
    df = pd.read_csv("../../data/interim/loans_clean.csv")

    Id = ['account_id']
    target = ['class']

    logger.info(f'Obtaining data types')
    binary_vars = [var for var in df.nunique()[df.nunique() == 2].keys() if var not in target]
    categorical_vars = [var for var in df.select_dtypes(include='object').columns.tolist() if
                        var not in binary_vars + target]
    numerical_vars = [var for var in df.columns if var not in Id + target + categorical_vars]

    logger.info(f'Encoding variables')
    le = LabelEncoder()
    for col in binary_vars:
        df[col] = le.fit_transform(df[col])

    df = pd.get_dummies(data=df, columns=categorical_vars)

    logger.info('Scaling numerical variables')
    scl = RobustScaler()
    scaled = scl.fit_transform(df[numerical_vars])
    scaled = pd.DataFrame(scaled, columns=numerical_vars)

    df = df.drop(columns=numerical_vars, axis=1)
    df = df.merge(scaled, left_index=True, right_index=True, how="left")

    df.to_csv(f'{output_file_path}loans_processed.csv', index=False)

    logger.info(f'Data exported to {output_file_path}')

    return


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
