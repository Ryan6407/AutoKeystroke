

class KeystrokeConfig():

    """
    Configuration class for data preparation and training. 
    """

    def __init__(self, 
                 min_burst_val=0.01,
                 window_values=[30.0, 60.0],
                 features=["id", "event_id", "down_time", "word_count", "cursor_position"],
                 pause_vals=[0.5, 1, 1.5, 2, 3],
                 ):
        self.min_burst_val = min_burst_val
        self.window_values = window_values
        self.features = features
        self.pause_vals = pause_vals