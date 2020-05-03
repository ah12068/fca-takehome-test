import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio

from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    LabelEncoder,
    RobustScaler
)

pio.templates.default = 'plotly_white'
pd.set_option('display.float_format', '{:.5f}'.format)

def main():
    logger = logging.getLogger(__name__)

    plot_output_path = f'../../reports/figures/'

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

    logger.info(f'Producing Correlation Plot')
    correlation = df[[var for var in df.columns if var not in Id + target]].corr()
    matrix_cols = correlation.columns.tolist()
    corr_array = np.array(correlation)

    trace = go.Heatmap(z=corr_array,
                       x=matrix_cols,
                       y=matrix_cols,
                       colorscale="Viridis",
                       colorbar=dict(title="Pearson Correlation coefficient",
                                     titleside="right"
                                     ),
                       )

    layout = go.Layout(dict(title="Correlation Matrix for variables",
                            title_x=0.5,
                            autosize=False,
                            height=720,
                            width=800,
                            margin=dict(r=0, l=210,
                                        t=25, b=210,
                                        ),
                            yaxis=dict(tickfont=dict(size=9)),
                            xaxis=dict(tickfont=dict(size=9))
                            )
                       )

    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    fig.write_html(f'{plot_output_path}corr_plot.html')
    fig.show()

    logger.info('Scaling numerical variables')
    scl = RobustScaler()
    scaled = scl.fit_transform(df[numerical_vars])
    scaled = pd.DataFrame(scaled, columns=numerical_vars)

    df = df.drop(columns=numerical_vars, axis=1)
    df = df.merge(scaled, left_index=True, right_index=True, how="left")

    logger.info(f'Performing PCA')
    pca = PCA(n_components=2)
    X = df[[i for i in df.columns if i not in target]]
    Y = df[target + Id]

    principal_components = pca.fit_transform(X)
    pca_data = pd.DataFrame(principal_components, columns=["PC1", "PC2"])
    pca_data = pca_data.merge(Y, left_index=True, right_index=True, how="left")

    sns.lmplot(
        data=pca_data,
        x="PC1",
        y="PC2",
        fit_reg=False,
        hue='class',
        legend=True
    )
    plt.savefig(f'{plot_output_path}pca_plot.png')

    logger.info(f'Plots saved to {plot_output_path}')

    return


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
