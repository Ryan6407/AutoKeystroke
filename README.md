# AutoKeystroke

#### Start off by importing the necessary packages you will need to process your data.

```python
from autokeystroke.modeling.trainer import KeystrokeTrainer
from autokeystroke.utils.config import KeystrokeConfig
```

#### Make sure you have your data formatted in a csv file with columns id, event_id, and down_time

```python
data_pth = "your_data_path.csv"
```

#### Instantiate a KeystrokeConfig and KeystrokeTrainer and adjust the parameters of the config according to your preferences and specifications

```python
config = KeystrokeConfig()
trainer = KeystrokeTrainer(data_pth, config)
```

#### Call the run method of the trainer instance and you will be able to find the trained models and results saved in the output directory

```python
trainer.run()
```
