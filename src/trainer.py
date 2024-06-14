from visuals.temporal_analysis import TemporalVisualizer
from visuals.base_analysis import BaseVisualizer
from utils.config import KeystrokeConfig
from utils.data_processing import DataHandler
import pandas as pd

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
        
        self.data_handler = DataHandler(self.raw_df, self.config)
        self.processed_df = self.data_handler.get_df()
        self.temporal_visualizer = TemporalVisualizer(self.raw_df)
        self.base_visualizer = BaseVisualizer(self.processed_df)

    def run(self):
        self.temporal_visualizer.create_user_visuals(self.config.user_samples)
        self.base_visualizer.create_visuals(self.processed_df.columns)

if __name__ == "__main__":
    trainer = KeystrokeTrainer("assets/train_logs.csv")
    trainer.run()
