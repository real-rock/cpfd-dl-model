import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def plot(df, cols=None):
    if cols == None:
        cols = df.columns
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    date_index = pd.date_range(start=df.index[0], end=df.index[-1], freq='1T')
    df = df.reindex(date_index)
    for col in cols:
        fig.add_trace(go.Scatter(x=df.index, y=df[col].values, name=col), secondary_y=(df[col].mean() < 2))
    fig.update_traces(connectgaps=False)
    fig.show()
