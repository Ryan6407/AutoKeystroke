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

class BaseVisualizer():
    """
    
    """
    def __init__(self, processed_df):
        self.df = processed_df