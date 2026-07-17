import os
from pprint import pprint

from utils import load_config, namespace_to_dict
from constants import TRAIN_CONFIGS_PATH

config_filepath = os.path.join(TRAIN_CONFIGS_PATH, "de-en-medium.yaml")
config = load_config(config_filepath)
pprint(namespace_to_dict(config))
print(f"Batch size: {config.trainer.batch_size}")