from lightgbm import LGBMRegressor

class KeystrokeConfig():

    """
    Configuration class for data preparation and training. 
    """

    def __init__(self, 
                 min_burst_val=0.01,
                 window_values=[30.0, 60.0],
                 features=["id", "event_id", "down_time", "word_count", "cursor_position"],
                 pause_vals=[0.5, 1, 1.5, 2, 3],
                 user_samples=3,
                 target_col=None,
                 validate=False,
                 split_size=None,
                 task="reg",
                 models=[LGBMRegressor()],
                 output_dir="output/",
                 target_path=None,
                 ):
        self.min_burst_val = min_burst_val
        self.window_values = window_values
        self.features = features
        self.pause_vals = pause_vals
        self.user_samples = user_samples
        self.target_col = target_col
        self.validate = validate
        self.models = models
        self.task = task 
        self.split_size = split_size
        self.output_dir = output_dir
        self.target_path = target_path