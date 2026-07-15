import argparse
import os
import torch
from torch import nn

from utils import load_config
from constants.constants import DATASET_PATHS, TRAINING_CONFIG_PATH

from src.tokenizer.tokenizer import Tokenizer
from src.model.enc_dec import EncoderDecoder

CONFIG_FILEPATH = None

def load_config_from_args():
    """Load configuration from command line arguments."""
    parser = argparse.ArgumentParser(description="Train transformer model")
    parser.add_argument("--config", type=str, help="Name of training run config file")
    args = parser.parse_args()

    # load the training config file
    global CONFIG_FILEPATH
    CONFIG_FILEPATH = os.path.join(TRAINING_CONFIG_PATH, f"{args.config}")
    config = load_config(CONFIG_FILEPATH)

    # load vocab_size from the tokenizer associated with the dataset specified by the training config file
    config.vocab_size = Tokenizer.load(os.path.join(DATASET_PATHS[config.dataset], "tokenizer.model")).vocab_size()

    # load block_size from the config file associated with the dataset specified by the training config file 
    ds_conf_path = os.path.join(DATASET_PATHS[config.dataset], "config.yaml")
    ds_conf = load_config(ds_conf_path)
    config.block_size = ds_conf.block_size

def create_model(
    config
) -> torch.nn.Module:
    model = EncoderDecoder(config)
    return model

def main():
    config = load_config_from_args()

    model = create_model(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model created with {param_count:,} parameters")

    memory_req = param_count * 2 * 4

if __name__ == "__main__":
    main()