import pickle as pkl
from numpy import round
import plotly.graph_objs as go
import plotly.io as pio

pio.templates.default = 'plotly_white'

models = [
    'LogisticRegression',
    'HuberRegression',
    'DecisionTreeClassifier'
]

vars = f'allvar'

fig = go.Figure()

for model in models:
    file = f'{model}_RandomOverSampler_{vars}_metrics.pickle'
    path = f'../../models/' + file
    metrics = pkl.load(open(path, 'rb'))
    fpr = metrics['FPR']
    tpr = metrics['TPR']
    model_roc_auc = metrics['AUC Score']

    trace1 = go.Scatter(x=fpr, y=tpr,
                        name=f'{model} AUC: {round(model_roc_auc)}',
                        line=dict(width=2))

    fig.add_trace(trace1)

trace2 = go.Scatter(x=[0, 1], y=[0, 1],
                    name='Random',
                    line=dict(color='rgb(205, 12, 24)', width=2,
                              dash='dot'))
fig.add_trace(trace2)

fig['layout'].update(
    showlegend=True,
    title=f"Model performance",
    title_x=0.5,
    autosize=False, height=900, width=800,
    plot_bgcolor='rgba(240,240,240, 0.95)',
    paper_bgcolor='rgba(240,240,240, 0.95)',
    margin=dict(b=195))

fig["layout"]["xaxis"].update(dict(title="false positive rate"))
fig["layout"]["yaxis"].update(dict(title="true positive rate"))
fig.layout['hovermode'] = 'x'

fig.write_html(f'../../reports/figures/model_RandomOverSampler_{vars}_comparison.html')
fig.show()
