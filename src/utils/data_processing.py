import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
import math
from scipy.spatial import distance
import pandas as pd
from typing import List
from config import KeystrokeConfig
from tqdm import tqdm 
tqdm.pandas()

def get_time_series_slope(time_series : List[int]) -> float:

    """
    Fits a line of best fit y=mx+b to the time series data
    and then extracts and returns the slope. Reveals the general 
    trend of the time series.
    """

    x_values = np.array(list(range(1, len(time_series)+1)))

    lr = LinearRegression()

    lr.fit(x_values.reshape(-1, 1), time_series)
    
    return lr.coef_[0]

def get_ar_params(time_series: List[int]) -> float:

    """
    Fits an autoregressive model to the time series
    and extracts it's parameters according to the formula
    y_t=δ+ϕ_1y_(t−1)+…+ϕ_py_(t−p)+ϵ_t.
    """

    model = AutoReg(time_series, lags=1)

    trained_model = model.fit()

    return trained_model.params


def get_shannon_entropy(time_series: List[int]) -> float:

    """
    Calculates the shannon entropy for a given time series
    Quantifies the amount of information contained within a variable
    """

    entropy_data = [x/sum(time_series) for x in time_series]

    s_entropy = 0

    for p in entropy_data:
        if p > 0:
            s_entropy += p * math.log(p, 2)

    return -s_entropy

def get_shannon_jensen_div(time_series: List[int]) -> float:

    """
    Calculates the shannon jensen divergence between the time series 
    scaled down between 0 and 1 compared to a uniform distribution.
    """

    time_series = np.array([x/sum(time_series) for x in time_series])

    uniform_dist = np.array([1/len(time_series) for x in time_series])

    return distance.jensenshannon(time_series, uniform_dist)

def get_num_changes(time_series: List[int]) -> float:

    """
    Counts the number of times a given time series experiences 
    changes from increasing to decreasing or decreasing to increasing.
    For example, the time series [1, 3, 2, 4, 1] -> 3
    """

    diffs = [time_series[i+1] - time_series[i] for i in range(len(time_series)-1)]
    extreme_count = 0

    for i in range(len(diffs)-1):
        if (diffs[i] < 0 and diffs[i+1] < 0) or (diffs[i] > 0 and diffs[i+1] > 0):
            pass
        else:
            extreme_count += 1

    return extreme_count


def get_avg_recurrence(time_series: List[int]) -> float:

    """
    Calculates the average length of each typing session 
    by determining the average length of time slices in which
    at least one keystroke event was made
    """

    total = 0
    count = 0
    values = []

    for idx in range(0, len(time_series)):
        if time_series[idx] == 0:
            count += 1
            values.append(total)
            total = 0
        else:
            total += 1
        
    if (time_series[len(time_series)-1] != 0):
        values.append(total)
        count += 1
            
    return sum(values) / count

def get_stddev_recurrence(time_series: List[int]) -> float:

    """
    Calculates the standard deviation of the length of typing sessions
    in which at least one keystroke event was made
    """

    total = 0
    count = 0
    values = []

    for idx in range(0, len(time_series)):
        if time_series[idx] == 0:
            count += 1
            values.append(total)
            total = 0
        else:
            total += 1
        
    if (time_series[len(time_series)-1] != 0):
        values.append(total)
        count += 1
            
    return np.std(values)

def get_cursor_back(time_series: List[int]) -> int:

    """
    Counts the number of times the cursor went backwards
    """

    return len([x for x in time_series if x < 0])

def count_bursts(data: List[int], min_burst_val=0.01) -> int:

    """
    Measures how many periods of times there were where 
    the difference between keystroke events were less than 
    the min_burst_val. Also measures the average length of the 
    pauses between bursts
    """

    burst_count = 0

    burst_start = 0

    pause_total = 0

    pause_count = 0

    bursts = []

    for i in range(1, len(data)):
        if data[i] > min_burst_val:
            pause_total += data[i]
            pause_count += 1
            if i - burst_start - 1 > 0:
                burst_count += 1
                bursts.append(data[i])

            burst_start = i

    if (burst_start != len(data)-1):
        burst_count += 1

    return [burst_count, pause_total/pause_count]

class DataHandler():
    """
    Main data processor class. Expects a dataframe with the following columns:  

    Required:
    id: the session of writing a particular keystroke belongs to
    event id: the order in which a particular keystroke occured relative to the typing session
    down time/up time: the time when a key was pressed and released

    Recommended:
    event: the type of event occuring for a keystroke (if data is anonymized, then any key press defaults to some filler letter e.g. q) 
    cursor_position: the location of the cursor
    """

    def __init__(self, df, config, target_column=None):
        self.df = df
        self.target_column = target_column
        self.config = config
    
    def feature_engineer(self) -> pd.DataFrame:

        """
        For a raw keystroke log dataframe, we create a new dataframe suitable for supervised learning tasks.
        Created features include 
        """

        self.df.sort_values(['id', 'up_time'], inplace=True)

        self.df['diffs'] = self.df.groupby(['id'])['up_time'].transform(lambda x: x.diff())

        self.df.sort_index(inplace=True)
        self.df["diffs_seconds"] = self.df["diffs"] / 1000

        self.df["time_elapsed"] = self.df.groupby("id")["diffs_seconds"].cumsum()
        self.df["time_elapsed"].fillna(0, inplace=True)

        new_df = pd.DataFrame({"id" : list(self.df.groupby("id").groups.keys())})

        for i in range(len(self.config.pause_vals[:-1])):
            pause_counts = self.df[(self.df["diffs_seconds"] > self.config.pause_vals[i]) & (self.df["diffs_seconds"] < self.config.pause_vals[i+1])].groupby("id").size()
            for id_ in new_df["id"].values:
                if pause_counts.get(id_) is None:
                    pause_counts[id_] = 0

            new_df[f"pause_{self.config.pause_vals[i]}"] = pause_counts.values

        pause_counts = self.df[self.df["diffs_seconds"] > self.config.pause_vals[-1]].groupby("id").size()

        for id_ in new_df["id"].values:
            if pause_counts.get(id_) is None:
                pause_counts[id_] = 0

        new_df[f"pause_{self.config.pause_vals[-1]}"] = pause_counts.values

        print(new_df)

        df_grouped = self.df.groupby("id")

        burst_vals = np.array(df_grouped.progress_apply(lambda x: count_bursts(x["diffs_seconds"].values)).values.tolist())

        new_df["mean_pause_duration"] = burst_vals[:,0]
        new_df["burst_count"] = burst_vals[:,1]

        new_df["verbosity"] = df_grouped.size().values
        backspace_df = self.df.groupby(["up_event", "id"]).size()["Backspace"]
        new_df = pd.merge(new_df, backspace_df.rename("backspaces"), on="id", how="left")

        new_df["word_count"] = df_grouped["word_count"].last().values

        period_df = self.df.groupby(["up_event", "id"]).size()["."]
        new_df = pd.merge(new_df, period_df.rename("sent_count"), on="id", how="left")

        enter_df = self.df.groupby(["up_event", "id"]).size()["Enter"]
        new_df = pd.merge(new_df, enter_df.rename("paragraph_count"), on="id", how="left")

        nonprod_df = self.df.groupby(["activity", "id"]).size()["Nonproduction"]
        new_df = pd.merge(new_df, nonprod_df.rename("Nonproduction"), on="id", how="left")

        new_df["avg_keystroke_speed"] = new_df["verbosity"] / df_grouped["time_elapsed"].tail(1).values

        for window in self.config.window_values:
            self.df[f"window_{window}_sec_idx"] = self.df["time_elapsed"].apply(lambda x: math.floor(x / window))
            ar = np.array(df_grouped.progress_apply(lambda x: get_ar_params(x[f"window_{window}_sec_idx"].value_counts().reindex(range(max(x[f"window_{window}_sec_idx"])+1), fill_value=0))).values.tolist())
            ar_p1 = ar[:,0]
            ar_p2 = ar[:,1]
            new_df[f"ar_{window}_p1"] = ar_p1
            new_df[f"ar_{window}_p2"] = ar_p2

            new_df[f"Slope_Degree_{window}"] = df_grouped.progress_apply(lambda x: get_time_series_slope(x[f"window_{window}_sec_idx"].value_counts().reindex(range(max(x[f"window_{window}_sec_idx"])+1), fill_value=0))).values
            new_df[f"Entropy_{window}"] = df_grouped.progress_apply(lambda x: get_shannon_entropy(x[f"window_{window}_sec_idx"].value_counts().reindex(range(max(x[f"window_{window}_sec_idx"])+1), fill_value=0))).values
            new_df[f"Degree_Uniformity_{window}"] = df_grouped.progress_apply(lambda x: get_shannon_jensen_div(x[f"window_{window}_sec_idx"].value_counts().reindex(range(max(x[f"window_{window}_sec_idx"])+1), fill_value=0))).values
            new_df[f"Local_Extremes_{window}"] = df_grouped.progress_apply(lambda x: get_num_changes(x[f"window_{window}_sec_idx"].value_counts().reindex(range(max(x[f"window_{window}_sec_idx"])+1), fill_value=0))).values
            new_df[f"Average_Recurrence_{window}"] = df_grouped.progress_apply(lambda x: get_avg_recurrence(x[f"window_{window}_sec_idx"].value_counts().reindex(range(max(x[f"window_{window}_sec_idx"])+1), fill_value=0))).values
            new_df[f"StdDev_Recurrence_{window}"] = df_grouped.progress_apply(lambda x: get_stddev_recurrence(x[f"window_{window}_sec_idx"].value_counts().reindex(range(max(x[f"window_{window}_sec_idx"])+1), fill_value=0))).values

        new_df["backspaces"].fillna(0, inplace=True)

        new_df["largest_latency"] = df_grouped["diffs"].max().values

        new_df["smallest_latency"] = df_grouped["diffs"].min().values

        new_df["median_latency"] = df_grouped["diffs"].median().values

        new_df["first_pause"] = self.df.groupby("id").diffs_seconds.first().values

        if "cursor_position" in self.config.features:
            self.df['curpos_diffs'] = self.df.groupby(['id'])['cursor_position'].transform(lambda x: x.diff())
            new_df["Cursor_Back_Count"] = df_grouped.progress_apply(lambda x: get_cursor_back(x["curpos_diffs"])).values

        if "word_count" in self.config.features:
            self.df['word_diffs'] = self.df.groupby(['id'])['word_count'].transform(lambda x: x.diff())
            new_df["Word_Back_Count"] = df_grouped.progress_apply(lambda x: get_cursor_back(x["word_diffs"])).values
            new_df["largest_insert"] = df_grouped["word_diffs"].max().values
            new_df["largest_delete"] = df_grouped["word_diffs"].min().values

        return new_df 
    
    def get_df(self):
        return self.feature_engineer()

df = pd.read_csv("assets/train_logs.csv")
handler = DataHandler(df, KeystrokeConfig())
new_df = handler.get_df()
print(new_df)