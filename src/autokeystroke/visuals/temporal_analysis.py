import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import iplot
from sklearn.linear_model import LinearRegression
from typing import List
import numpy as np
import random
import math

def view_keystrokes_per_time(num_keystrokes : List[int], width=1800, height=700, user_id=""):

    """
    Creates a graph of the number of keystrokes occuring in a certain slice of time for all slices
    in the typing session. Additionally, a line of best fit is plotted so we can see the general trend.

    :param list num_keystrokes: The series of keystrokes per time slice
    :param int width: the width of the graph
    :param int height: the height of the graph
    """

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
            title=f"Keystroke Verbosity Per Time Slice for User {user_id}",
            xaxis_title="Time Slice",
            yaxis_title="Frequency",
            width=width,
            height=height,
        )

    iplot(fig)

def view_time_per_keystroke(keystroke_diffs : List[float], width=1800, height=700, user_id=""):

    """
    Creates a graph of the time differences between keystrokes in a typing session. 

    :param list keystroke_diffs: The series of differences between subsequent keystroke events
    :param int width: the width of the graph
    :param int height: the height of the graph
    """

    fig = make_subplots()
  
    trace1 = go.Scatter(
        x=list(range(1, len(keystroke_diffs) + 1)),
        y=keystroke_diffs,
        name='Time Spent Between Next Keystroke',
    )

    fig.add_trace(trace1)

    fig.update_layout(
            title=f"Time Spent Between Keystroke for User {user_id}",
            xaxis_title="Keystroke Event Number",
            yaxis_title="Time (seconds)",
            width=width,
            height=height,
        )

    iplot(fig)

class TemporalVisualizer():

    """
    Primary class for visualizing keystroke analytics
    This class creates graphics for time based relationships
    """

    def __init__(self, keystroke_df):
        self.df = keystroke_df
        self.df.sort_values(['id', 'up_time'], inplace=True)

        self.df['diffs'] = self.df.groupby(['id'])['up_time'].transform(lambda x: x.diff())

        self.df.sort_index(inplace=True)
        self.df["diffs_seconds"] = self.df["diffs"] / 1000

        self.df["time_elapsed"] = self.df.groupby("id")["diffs_seconds"].cumsum()
        self.df["time_elapsed"].fillna(0, inplace=True)

    def _sample_user(self) -> pd.DataFrame:

        """
        Samples data from a particular users typing session
        """

        rand_idx = random.randint(0, len(self.df["id"].unique())-1)
        return self.df[self.df["id"] == self.df["id"].unique()[rand_idx]]
    
    def create_user_visuals(self, num_users : int = 1):

        """
        Creates visuals for a particular users typing session

        :param int num_users: number of users to sample and visualize
        """

        for i in range(num_users):
            user_data = self._sample_user()
            user_data[f"window_30_sec_idx"] = user_data["time_elapsed"].apply(lambda x: math.floor(x / 30))
            view_keystrokes_per_time(user_data.groupby("window_30_sec_idx").size().values, user_id=user_data["id"].values[0])
            view_time_per_keystroke(user_data["diffs_seconds"].values, user_id=user_data["id"].values[0])