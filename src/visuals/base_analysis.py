import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import iplot
from sklearn.linear_model import LinearRegression
from typing import List
import numpy as np

class BaseVisualizer():
    """
    Creates graphs that illustrate general patterns that describe the entire dataset and 
    all of its users. 
    """
    def __init__(self, processed_df):
        self.df = processed_df

    def hist_feature_visualizer(self, column : str):

        """
        Creates histogram graphs for a specified column
        :param str column: column to be graphed
        """

        fig = px.histogram(self.df, x=column)
        iplot(fig)

    def create_visuals(self, features : List[str]):
        
        """
        Creates histograms for all of the specified columns
        :param list features: columns to be graphed
        """

        for f in features:
            if "id" not in f:
                self.hist_feature_visualizer(f)