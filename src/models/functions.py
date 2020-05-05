import logging
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from constants import (
    PLOTLY_TEMPLATE,
    PANDAS_TEMPLATE,
    classes,
    id_col,
    target_col,
    parameter_grid,
    cpu_count,
    feature_coefs,
    sampling_strats
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve


def baseline_trainer(processed_df, algorithm, sampler, cf, name=None):
    logger = logging.getLogger(__name__)

    cols = [col for col in processed_df.columns if col not in id_col + target_col]

    X = processed_df[cols]
    y = processed_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=111)

    logger.info(f'Applying sampling strategy')

    sampled_X, sampled_y = sampler.fit_sample(X_train, y_train)
    sampled_X = pd.DataFrame(data=sampled_X, columns=cols)
    sampled_y = pd.DataFrame(data=sampled_y, columns=target_col)

    logger.info(f'Fitting and Optimising Model')

    clf = GridSearchCV(
        estimator=algorithm,
        param_grid=parameter_grid[name],
        cv=10,
        n_jobs=cpu_count,
        verbose=True
    )

    best_model = clf.fit(sampled_X.values, sampled_y.values.ravel())
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)

    logger.info(f'Generating metrics')

    if cf == "coefficients":
        coefficients = pd.DataFrame(best_model.best_estimator_.coef_.ravel())
        plotly_title = 'Coefficients'
    elif cf == "features":
        coefficients = pd.DataFrame(best_model.best_estimator_.feature_importances_)
        plotly_title = 'Feature Importances'

    column_df = pd.DataFrame(cols)
    coef_sumry = (pd.merge(coefficients, column_df, left_index=True,
                           right_index=True, how="left"))
    coef_sumry.columns = ["coefficients", "features"]
    coef_sumry = coef_sumry.sort_values(by="coefficients", ascending=False)

    conf_matrix = confusion_matrix(y_test, predictions)
    model_roc_auc = roc_auc_score(y_test, predictions)
    fpr, tpr, thresholds = roc_curve(y_test, probabilities[:, 1])

    metrics = {
        'Classification Report': classification_report(y_test, predictions),
        'AUC Score': model_roc_auc,
        'Confusion Matrix': conf_matrix,
        'TPR': tpr,
        'FPR': fpr
    }
    print(f"Metrics:\n{metrics['Classification Report']}\n{model_roc_auc}")

    logger.info('Producing Evaluation Report')

    trace1 = go.Heatmap(z=conf_matrix,
                        x=classes,
                        y=classes,
                        showscale=False,
                        colorscale="Picnic",
                        name="matrix")

    # plot roc curve
    trace2 = go.Scatter(x=fpr, y=tpr,
                        name=f'AUC: {model_roc_auc}',
                        line=dict(color='rgb(22, 96, 167)', width=2))
    trace3 = go.Scatter(x=[0, 1], y=[0, 1],
                        line=dict(color='rgb(205, 12, 24)', width=2,
                                  dash='dot'))

    # plot coeffs
    trace4 = go.Bar(x=coef_sumry["features"], y=coef_sumry["coefficients"],
                    name=cf,
                    marker=dict(color=coef_sumry["coefficients"],
                                colorscale="Picnic",
                                line=dict(width=.6, color="black")))

    # subplots
    fig = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                        subplot_titles=('Confusion Matrix',
                                        'Receiver operating characteristic',
                                        plotly_title))

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)
    fig.append_trace(trace3, 1, 2)
    fig.append_trace(trace4, 2, 1)

    fig['layout'].update(showlegend=False, title=f"{name} performance",
                         autosize=False, height=900, width=800,
                         plot_bgcolor='rgba(240,240,240, 0.95)',
                         paper_bgcolor='rgba(240,240,240, 0.95)',
                         margin=dict(b=195))
    fig["layout"]["xaxis2"].update(dict(title="false positive rate"))
    fig["layout"]["yaxis2"].update(dict(title="true positive rate"))
    fig["layout"]["xaxis3"].update(dict(showgrid=True, tickfont=dict(size=10),
                                        tickangle=90))
    fig.layout['hovermode'] = 'x'
    fig.show()

    return algorithm, metrics, fig


def feature_select(processed_df, sampler, algorithm, n_features):
    logger = logging.getLogger(__name__)

    cols = [col for col in processed_df.columns if col not in id_col + target_col]

    X = processed_df[cols]
    y = processed_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=111)

    sampled_X, sampled_y = sampling_strats[sampler].fit_sample(X_train, y_train)
    sampled_X = pd.DataFrame(data=sampled_X, columns=cols)
    sampled_y = pd.DataFrame(data=sampled_y, columns=target_col)

    logger.info(f'Performing RFE for {n_features} features')

    model = RFE(estimator=algorithm, n_features_to_select=n_features)
    model.fit(X_train.values, y_train.values.ravel())
    cols_to_keep = X.columns[model.support_].tolist()

    print(f'Top {n_features} Features to keep: {cols_to_keep}')

    return cols_to_keep


def train_with_feature_selection(processed_df, sampler, algorithm, n_features, cf, name=None):
    logger = logging.getLogger(__name__)

    features = feature_select(processed_df, sampler, algorithm, n_features)

    X = processed_df[features]
    y = processed_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=111)

    logger.info(f'Applying sampling strategy')

    sampled_X, sampled_y = sampling_strats[sampler].fit_sample(X_train, y_train)
    sampled_X = pd.DataFrame(data=sampled_X, columns=features)
    sampled_y = pd.DataFrame(data=sampled_y, columns=target_col)

    clf = GridSearchCV(
        estimator=algorithm,
        param_grid=parameter_grid[name],
        cv=10,
        n_jobs=cpu_count,
        verbose=True
    )

    best_model = clf.fit(sampled_X.values, sampled_y.values.ravel())
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)

    logger.info(f'Generating metrics')

    if cf == "coefficients":
        coefficients = pd.DataFrame(best_model.best_estimator_.coef_.ravel())
        plotly_title = 'Coefficients'
    elif cf == "features":
        coefficients = pd.DataFrame(best_model.best_estimator_.feature_importances_)
        plotly_title = 'Feature Importances'

    column_df = pd.DataFrame(features)
    coef_sumry = (pd.merge(coefficients, column_df, left_index=True,
                           right_index=True, how="left"))
    coef_sumry.columns = ["coefficients", "features"]
    coef_sumry = coef_sumry.sort_values(by="coefficients", ascending=False)

    conf_matrix = confusion_matrix(y_test, predictions)
    model_roc_auc = roc_auc_score(y_test, predictions)
    fpr, tpr, thresholds = roc_curve(y_test, probabilities[:, 1])

    metrics = {
        'Classification Report': classification_report(y_test, predictions),
        'AUC Score': model_roc_auc,
        'Confusion Matrix': conf_matrix,
        'TPR': tpr,
        'FPR': fpr
    }
    print(f"Metrics:\n{metrics['Classification Report']}\n{model_roc_auc}")

    logger.info('Producing Evaluation Report')

    trace1 = go.Heatmap(z=conf_matrix,
                        x=classes,
                        y=classes,
                        showscale=False,
                        colorscale="Picnic",
                        name="matrix")

    # plot roc curve
    trace2 = go.Scatter(x=fpr, y=tpr,
                        name=f'AUC: {model_roc_auc}',
                        line=dict(color='rgb(22, 96, 167)', width=2))
    trace3 = go.Scatter(x=[0, 1], y=[0, 1],
                        line=dict(color='rgb(205, 12, 24)', width=2,
                                  dash='dot'))

    # plot coeffs
    trace4 = go.Bar(x=coef_sumry["features"], y=coef_sumry["coefficients"],
                    name=cf,
                    marker=dict(color=coef_sumry["coefficients"],
                                colorscale="Picnic",
                                line=dict(width=.6, color="black")))

    # subplots
    fig = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                        subplot_titles=('Confusion Matrix',
                                        'Receiver operating characteristic',
                                        plotly_title))

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)
    fig.append_trace(trace3, 1, 2)
    fig.append_trace(trace4, 2, 1)

    fig['layout'].update(showlegend=False, title=f"{name} performance",
                         autosize=False, height=900, width=800,
                         plot_bgcolor='rgba(240,240,240, 0.95)',
                         paper_bgcolor='rgba(240,240,240, 0.95)',
                         margin=dict(b=195))
    fig["layout"]["xaxis2"].update(dict(title="false positive rate"))
    fig["layout"]["yaxis2"].update(dict(title="true positive rate"))
    fig["layout"]["xaxis3"].update(dict(showgrid=True, tickfont=dict(size=10),
                                        tickangle=90))
    fig.layout['hovermode'] = 'x'
    fig.show()

    return algorithm, metrics, fig


def create_report(algorithm, test_X, test_Y):
    predictions = algorithm.predict(test_X)
    probabilities = algorithm.predict_proba(test_X)

    model_roc_auc = roc_auc_score(test_Y, predictions)

    metrics = f'''
        Classification Report:\n{classification_report(test_Y, predictions)}\n,
        Accuracy Score : {accuracy_score(test_Y, predictions)}\n"
        AUC Score: {model_roc_auc}\n,
        Confusion Matrix:\n{confusion_matrix(test_Y, predictions)}\n
                '''

    return metrics


def plot_report(processed_df, algorithm, test_X, test_Y, cf, name=None):
    cols = [i for i in processed_df.columns if i not in id_col + target_col]

    if cf == "coefficients":
        coefficients = pd.DataFrame(algorithm.best_estimator_.bestcoef_.ravel())
    elif cf == "features":
        coefficients = pd.DataFrame(algorithm.best_estimator_.feature_importances_)

    column_df = pd.DataFrame(cols)

    coef_sumry = (pd.merge(coefficients, column_df, left_index=True,
                           right_index=True, how="left"))
    coef_sumry.columns = ["coefficients", "features"]
    coef_sumry = coef_sumry.sort_values(by="coefficients", ascending=False)

    predictions = algorithm.predict(test_X)
    probabilities = algorithm.predict_proba(test_X)
    conf_matrix = confusion_matrix(test_Y, predictions)
    model_roc_auc = roc_auc_score(test_Y, predictions)
    fpr, tpr, thresholds = roc_curve(test_Y, probabilities[:, 1])

    trace1 = go.Heatmap(z=conf_matrix,
                        x=classes,
                        y=classes,
                        showscale=False,
                        colorscale="Picnic",
                        name="matrix")

    # plot roc curve
    trace2 = go.Scatter(x=fpr, y=tpr,
                        name="Roc : " + str(model_roc_auc),
                        line=dict(color='rgb(22, 96, 167)', width=2))
    trace3 = go.Scatter(x=[0, 1], y=[0, 1],
                        line=dict(color='rgb(205, 12, 24)', width=2,
                                  dash='dot'))

    # plot coeffs
    trace4 = go.Bar(x=coef_sumry["features"], y=coef_sumry["coefficients"],
                    name="coefficients",
                    marker=dict(color=coef_sumry["coefficients"],
                                colorscale="Picnic",
                                line=dict(width=.6, color="black")))

    # subplots
    fig = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                        subplot_titles=('Confusion Matrix',
                                        'Receiver operating characteristic',
                                        'Coefficients / Feature Importances'))

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)
    fig.append_trace(trace3, 1, 2)
    fig.append_trace(trace4, 2, 1)

    fig['layout'].update(showlegend=False, title=f"{name} performance",
                         autosize=False, height=900, width=800,
                         plot_bgcolor='rgba(240,240,240, 0.95)',
                         paper_bgcolor='rgba(240,240,240, 0.95)',
                         margin=dict(b=195))
    fig["layout"]["xaxis2"].update(dict(title="false positive rate"))
    fig["layout"]["yaxis2"].update(dict(title="true positive rate"))
    fig["layout"]["xaxis3"].update(dict(showgrid=True, tickfont=dict(size=10),
                                        tickangle=90))
    fig.layout['hovermode'] = 'x'
    fig.show()

    return algorithm


def plot_roc_auc(algorithms, test_X, test_Y, name=None):
    fig = go.Figure()
    for algorithm in algorithms.keys():
        predictions = algorithms[algorithm].predict(test_X)
        probabilities = algorithms[algorithm].predict_proba(test_X)
        model_roc_auc = roc_auc_score(test_Y, predictions)
        fpr, tpr, thresholds = roc_curve(test_Y, probabilities[:, 1])

        # plot roc curve
        trace1 = go.Scatter(x=fpr, y=tpr,
                            name=f"{algorithm}, AUC: {model_roc_auc}",
                            line=dict(color='rgb(22, 96, 167)', width=2))
        trace2 = go.Scatter(x=[0, 1], y=[0, 1],
                            line=dict(color='rgb(205, 12, 24)', width=2,
                                      dash='dot'))
        fig.add_trace([trace1, trace2])

    fig['layout'].update(showlegend=False, title=f"{name} performance",
                         autosize=False, height=900, width=800,
                         plot_bgcolor='rgba(240,240,240, 0.95)',
                         paper_bgcolor='rgba(240,240,240, 0.95)',
                         margin=dict(b=195))
    fig["layout"]["xaxis"].update(dict(title="false positive rate"))
    fig["layout"]["yaxis"].update(dict(title="true positive rate"))

    fig.layout['hovermode'] = 'x'
    fig.show()

    return
