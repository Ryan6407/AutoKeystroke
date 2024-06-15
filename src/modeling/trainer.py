from visuals.temporal_analysis import TemporalVisualizer
from visuals.base_analysis import BaseVisualizer
from utils.config import KeystrokeConfig
from utils.data_processing import DataHandler
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import pickle
import json
import os

class KeystrokeTrainer():

    """
    Main class for keystroke modeling and analysis. Requires the path
    to the raw keystroke data. Additionally, one can pass in a custom configuration
    if different settings are desired. 
    """

    def __init__(self, data_pth : str, config : KeystrokeConfig = KeystrokeConfig()):
        self.raw_df = pd.read_csv(data_pth)
        self.columns = self.raw_df.columns
        self.config = config
        
        if "id" not in self.columns or "event_id" not in self.columns or "down_time" not in self.columns:
            raise ValueError("Must have all three required features ('id', 'event_id', 'down_time') in dataframe. Please rename these columns before passing in the csv path")
        
        print("-"*10 + "Processing Data" + "-"*10 + "\n")

        self.data_handler = DataHandler(self.raw_df, self.config)
        self.processed_df = self.data_handler.get_df()

        print(f"{len(self.processed_df)} samples of data created with {self.processed_df.shape[1]} features\n")

        print("-"*10 + "Feature Descriptions" + "-"*10 + "\n")
        print("""
        Pause value(s): The number of pauses during the writing session that by duration were in between the value specified and the next
        value specified
        Mean pause duration: the average length of pauses between bursts in seconds
        Burst count: the number of bursts during the typing session
        Verbosity: total number of keystrokes in the typing session
        Backspace count: total number of backspaces occuring during the typing session
        Word count: total number of words typed out during the typing session
        Sent count: total number of sentences during the typing session
        Paragraph count: total number of paragraphs during the typing session
        Nonproduction: total number of keystrokes not contributing to the overall text
        Average keystroke speed: average amount of time spent on each keystroke
        AR P1/P2: The two parameters of an autoregressive model fit to the number of keystrokes occuring over time within varying time intervals 
        controlled by the user in the config
        Slope degree: the slope of a linear model fit to the number of keystrokes occuring over time within varying time intervals 
        controlled by the user in the config
        Entropy: the shannon entropy of the number of keystrokes occuring over time within varying time intervals 
        controlled by the user in the config
        Degree uniformity: the shannon-jensen divergence of the number of keystrokes occuring over time within varying time intervals 
        controlled by the user in the config
        Local extremes: the number of local highs/lows occuring in the number of keystrokes occuring over time within varying time intervals 
        controlled by the user in the config
        Average recurrence: the average length of consecutive time slices where at least one keystroke event occured
        Stddev recurrence: the standard deviation of lengths of consecutive time slices where at least one keystroke event occured
        Largest latency: the biggest difference in time between two keystrokes
        Median Latency: the median of differences between keystrokes
        Smallest latency: the smallest difference between keystrokes
        First pause: the initial pause at the start of the typing session
        Cursor Back: the number of times during the typing session where the cursor moved backwards
        Word Back: the number of words that were deleted during the typing session
        Largest insert: the greatest number of words pasted into the text
        Largest delete: the greatest number of words removed from the text
        """)

        if self.config.target_col is not None:
            target_values = pd.read_csv(self.config.target_path)
            self.processed_df = pd.merge(self.processed_df, target_values, on="id")

        self.temporal_visualizer = TemporalVisualizer(self.raw_df)
        self.base_visualizer = BaseVisualizer(self.processed_df)

        self.results = {"results" : []}

    def train(self, estimator, train_df : pd.DataFrame, targets : pd.Series):
        """
        Trains an estimator on the processed data and returns the trained
        estimator instance. Expects a model with scikit-learn syntax (LightGBM,
        XGBoost, CatBoost supported)
        """
        if not os.path.exists(f"{self.config.output_dir}"):
            os.makedirs(f"{self.config.output_dir}")

        if self.config.validate:

            from sklearn.model_selection import train_test_split
            X_train, X_valid, y_train, y_valid = train_test_split(train_df, targets.values, test_size=self.config.split_size, random_state=2007, shuffle=True)

            estimator.fit(X_train, y_train)

            with open(f"{os.path.abspath(self.config.output_dir + estimator.__class__.__name__)}.pkl", "wb") as f:
                pickle.dump(estimator, f)

            valid_pred = estimator.predict(X_valid)

            if self.config.task == "reg":
                from sklearn.metrics import r2_score, mean_squared_error
                self.results["results"].append(
                    {
                        "Model Name" : estimator.__class__.__name__,
                        "Metrics" : 
                                [
                                    ("Mean Squared Error", mean_squared_error(y_valid, valid_pred)),
                                    ("R2 Score", r2_score(y_valid, valid_pred)),
                                ], 
                    }
                )
            
            elif self.config.task == "cls":

                from sklearn.metrics import log_loss, accuracy_score

                valid_proba = estimator.predict_proba(X_valid)

                self.results["results"].append(
                    {
                        "Model Name" : estimator.__class__.__name__,
                        "Metrics" : 
                                [
                                    ("Log Loss", log_loss(y_valid, valid_proba)),
                                    ("Accuracy", accuracy_score(y_valid, valid_pred)),
                                ]
                    }
                )       
        else:
            estimator.fit(train_df, targets.values)
            with open(f"{self.config.output_dir}{estimator.__class__.__name__}.pkl") as f:
                pickle.dump(estimator, f)

    def run(self):

        print("-"*10 + "Performing Data Analysis" + "-"*10 + "\n")

        print(f"{2 * self.config.user_samples + len(self.processed_df.columns)} graphs created\n")

        self.temporal_visualizer.create_user_visuals(self.config.user_samples)
        self.base_visualizer.create_visuals(self.processed_df.columns)

        if self.config.target_col is None:

            print("*"*20, "Keystroke graphics provided\nNo modeling performed as the target column was None", "*"*20)
        
        else:

            targets = self.processed_df.pop(self.config.target_col)

            for estimator in self.config.models:
                self.train(estimator, self.processed_df.drop("id", axis=1), targets)

            for result in self.results["results"]:
                print("\n\n")
                print("-"*10 + f"{result['Model Name']} Training Results" + "-"*10 + "\n")
                for metric in result["Metrics"]:
                    print(f"{metric[0]} : {metric[1]}\n")

        
        with open(f'{self.config.output_dir}/results.json', 'w') as fp:
            json.dump(self.results, fp)
    
