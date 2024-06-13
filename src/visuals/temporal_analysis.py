import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import iplot
from sklearn.linear_model import LinearRegression
from typing import List
import numpy as np

def view_keystrokes_per_time(num_keystrokes : List[int], width=1400, height=700):

    x_vals = list(range(1, len(num_keystrokes)+1))
    x_vals = np.array(x_vals).reshape(-1, 1)

    lr = LinearRegression()
    lr.fit(x_vals, num_keystrokes)
    best_fit = lr.predict(x_vals)
        
    fig = make_subplots()

    trace1 = go.Scatter(
        x=x_vals.squeeze(),
        y=num_keystrokes,
        name='# Keystrokes Per Time Slice',
    )

    trace2 = go.Line(
        x=x_vals.squeeze(),
        y=best_fit,
        name="Best Fit"
    )

    fig.add_trace(trace1)
    fig.add_trace(trace2)  

    fig.update_layout(
            title="Keystroke Verbosity Per Time Slice",
            xaxis_title="Time Slice",
            yaxis_title="Frequency",
            width=width,
            height=height,
        )

    iplot(fig)

def view_time_per_keystroke(keystroke_diffs : List[float], width=1400, height=700):

    fig = make_subplots()
  
    trace1 = go.Scatter(
        x=list(range(1, len(keystroke_diffs) + 1)),
        y=keystroke_diffs,
        name='Time Spent Between Next Keystroke',
    )

    fig.add_trace(trace1)

    fig.update_layout(
            title="Time Spent Between Keystroke",
            xaxis_title="Keystroke Event Number",
            yaxis_title="Time (seconds)",
            width=width,
            height=height,
        )

    iplot(fig)

class KeystrokeVisualizer():
    def __init__(self, keystroke_df):
        self.keystroke_df = keystroke_df


view_keystrokes_per_time([200, 30, 10, 120, 312, 392, 203])