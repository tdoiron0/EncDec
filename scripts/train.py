import argparse
import logging
import os
import sys
import time
import yaml
import shutil
from dataclasses import dataclass, asdict
from typing import List, Callable

import torch
from torch import nn

from src.model.trainer import Trainer
from src.model.enc_dec import EncoderDecoder
from src.tokenizer.tokenizer import Tokenizer
from constants.constants import PROJECT_ROOT, DATASET_PATHS, RUNS_PATH, TRAINING_CONFIG_PATH
from constants.data_index import DATASET_INDEX

from scripts.evaluate import evaluate_model, load_tokenizer, load_eval_pairs
from utils import get_best_device, load_config

# TODO Claude told me to add this to reduce compute. Figure out what it does
torch.set_float32_matmul_precision("high")  # TF32 for fp32 matmuls
torch.backends.cudnn.allow_tf32 = True       # harmless here (no convs), but standard

device = get_best_device()

RUN_DIR = None
CONFIG_FILEPATH = None

def make_run_dir(config) -> str:
    """Create a unique run directory under runs/, appending a counter when the name is taken."""
    counter = 1
    while True:
        run_dir = os.path.join(RUNS_PATH, f"{config.run_name}-{counter}")
        if not os.path.exists(run_dir):
            break
        counter += 1
    os.makedirs(run_dir)
    return run_dir

def get_run_dir(config):
    if RUN_DIR is None:
        return make_run_dir(config)
    else:
        return RUN_DIR

def setup_logging(run_dir: str) -> None:
    """Configure logging for training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(os.path.join(run_dir, "training.log")), logging.StreamHandler()],
    )

def create_model(
    config
) -> torch.nn.Module:
    """Create and configure the EncoderDecoder model."""
    logger = logging.getLogger(__name__)

    model = EncoderDecoder(config)
    logger.info(
        f"Created EncoderDecoder model with {config.n_layer} layers/stack, "
        f"{config.n_embd} embedding dim, {config.n_head} heads"
    )
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {param_count:,} parameters")
    return model

def create_save_function(config) -> Callable:
    def save_checkpoint(trainer, announce=True):
        run_dir = get_run_dir(config)
        checkpoint_path = os.path.join(run_dir, config.checkpoint_dir)
        os.makedirs(checkpoint_path, exist_ok=True)

        torch.save(
            {
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "epoch": trainer.epoch,
                "config": config
            },
            os.path.join(checkpoint_path, f"checkpoint_epoch_{trainer.epoch}_iter_{trainer.iter_num}.pt"),
        )
        
        if announce:
            logging.info(f"Checkpoint saved to {checkpoint_path}")

    return save_checkpoint

def create_log_function(config) -> Callable:
    """Create the logging function for training progress."""
    logger = logging.getLogger(__name__)

    def log_training_progress(trainer):
        time_left = (((trainer.total_samps*config.max_epochs) - (trainer.samps + trainer.total_samps*trainer.epoch)) / trainer.rate) / 60
        if config.device == "cuda":
            message = (
                f"epoch={trainer.epoch}: "
                f"{trainer.samps}/{trainer.total_samps} samps "
                f"({trainer.rate:.2f} samp/sec) | "
                f"train loss {trainer.loss.item():.5f} | "
                f"time remaining: {time_left:.2f} min | "
                f"alloc {torch.cuda.max_memory_allocated()/1e9:.2f} GB | "
                f"reserved {torch.cuda.memory_reserved()/1e9:.2f} GB"
            )
            torch.cuda.reset_peak_memory_stats()
        else:
            message = (
                f"epoch={trainer.epoch}: "
                f"{trainer.samps}/{trainer.total_samps} samps "
                f"({trainer.rate:.2f} samp/sec) | "
                f"train loss {trainer.loss.item():.5f} | "
                f"time remaining {time_left:.2f} min | "
                f"pad waste {trainer.pad_waste}"
            )

        logger.info(message)

    return log_training_progress

def train_model(
    config,
    model: torch.nn.Module
) -> tuple:
    """Train the model with the given configuration."""
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

    config.device = device

    # Create dataset
    logger.info("Loading dataset")
    train_ds = DATASET_INDEX[config.dataset]("train", config.block_size)
    logger.info(f"Created training dataset with {len(train_ds)} samples")

    # Create logging + per-epoch validation functions
    log_fn = create_log_function(config)
    save_fn = create_save_function(config)

    # Create trainer with logging + validation hooks
    trainer = Trainer(config, model, train_ds, log_fn=log_fn, save_fn=save_fn)

    # Start training
    trainer.run()

    logger.info("Training completed!")
    return model, trainer

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

    return config

def main():
    """Main training function."""
    config = load_config_from_args()

    global RUN_DIR
    RUN_DIR = get_run_dir(config)

    # copy config file to run dir
    shutil.copy(CONFIG_FILEPATH, os.path.join(RUN_DIR, "train_config.yaml"))
    shutil.copy(os.path.join(DATASET_PATHS[config.dataset], "tokenizer.model"), os.path.join(RUN_DIR, "tokenizer.model"))

    # Setup
    setup_logging(RUN_DIR)
    logger = logging.getLogger(__name__)
    logger.info("Starting transformer training script")

    # Load configuration
    logger.info(f"Training configuration loaded")
    logger.info(f"Config file name: {config.run_name}")
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Max epochs: {config.max_epochs}")
    logger.info(f"Using device: {device}")

    try:
        model = create_model(config)

        model, trainer = train_model(config, model)

        logger.info("Training script completed successfully!")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
