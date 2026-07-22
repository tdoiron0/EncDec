import argparse
import logging
import os
import psutil
import sys
import time
import yaml
import shutil
import torch
import pprint

from dataclasses import dataclass, asdict
from typing import List, Callable
from git import Repo
from torch import nn

import index as index
from src.trainer.trainer import Trainer
from src.model.enc_dec import EncoderDecoder
from src.tokenizer.tokenizer import Tokenizer
from constants import PROJECT_ROOT, DATASET_DIRS, RUNS_PATH, TRAIN_CONFIGS_PATH

from scripts.evaluate import evaluate_model, load_tokenizer, load_eval_pairs
from utils import get_best_device, load_config, resolve, namespace_to_dict

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

device = get_best_device()

RUN_DIR = None

def run_name_taken(name) -> bool:
    return os.path.exists(os.path.join(RUNS_PATH, name))

def setup_logging(run_dir: str) -> None:
    """Configure logging for training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(os.path.join(run_dir, "training.log")), logging.StreamHandler()],
    )

def create_model(config) -> torch.nn.Module:
    logger = logging.getLogger(__name__)

    logger.info("Initializing model")
    # the model resolves vocab_size from the dataset's tokenizer, so it needs
    # to know which dataset it is being trained on
    config.model.dataset = config.dataset
    model = index.MODEL_INDEX[config.model.name](config.model)

    freeze_patterns = getattr(config.trainer, "freeze", None) or []
    if freeze_patterns:
        frozen = model.freeze_parameters(freeze_patterns)
        logger.info(f"Froze {len(frozen)} parameter tensors matching {freeze_patterns}")

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {param_count:,} parameters ({trainable_count:,} trainable)")
    return model

def create_trainer(config) -> Trainer:
    logger = logging.getLogger(__name__)
    logger.info("Initializing trainer")
    trainer = index.TRAINER_INDEX[config.trainer.name]()
    logger.info(f"Created trainer {config.trainer.name}")
    return trainer

def create_dataset(config) -> torch.utils.data.Dataset:
    logger = logging.getLogger(__name__)
    logger.info("Loading dataset")
    ds = index.DATASET_INDEX[config.dataset]("train")
    logger.info(f"Created training dataset with {len(ds)} samples")
    return ds

def create_optimizer(config, model) -> torch.optim.Optimizer:
    logger = logging.getLogger(__name__)
    logger.info("Initializing optimizer")
    optimizer = index.OPTIMIZER_INDEX[config.optimizer.name](config, model)
    logger.info(f"Initialized optimizer {config.optimizer.name}")
    return optimizer

def create_scheduler(config, optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    logger = logging.getLogger(__name__)
    logger.info(f"Initializing scheduler")
    scheduler = index.SCHEDULER_INDEX[config.scheduler.name](config, optimizer)
    logger.info(f"Initialized scheduler {config.scheduler.name}")
    return scheduler

def create_save_function(config) -> Callable:
    def save_checkpoint(trainer, announce=True):
        global RUN_DIR
        run_dir = RUN_DIR
        out_path = os.path.join(run_dir, config.checkpoint_dir)
        os.makedirs(out_path, exist_ok=True)

        torch.save(
            {
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "epoch": trainer.epoch
            },
            os.path.join(out_path, f"checkpoint_epoch_{trainer.epoch}_iter_{trainer.iter_num}.pt"),
        )
        
        if announce:
            logging.info(f"Checkpoint saved to {out_path}")

    return save_checkpoint

def create_log_function(config) -> Callable:
    logger = logging.getLogger(__name__)

    def log_training_progress(trainer):
        time_left = (((trainer.total_samps*config.trainer.max_epochs) - (trainer.samps + trainer.total_samps*trainer.epoch)) / trainer.rate) / 60
        if device == "cuda":
            message = (
                f"epoch={trainer.epoch}: "
                f"{trainer.samps}/{trainer.total_samps} samps "
                f"({trainer.rate:.2f} samp/sec) | "
                f"train loss: {trainer.loss.item():.5f} | "
                f"time remaining: {time_left:.2f} min | "
                f"alloc: {torch.cuda.max_memory_allocated()/1e9:.2f} GB | "
                f"reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB"
            )
            torch.cuda.reset_peak_memory_stats()
        else:
            message = (
                f"epoch={trainer.epoch}: "
                f"{trainer.samps}/{trainer.total_samps} samps "
                f"({trainer.rate:.2f} samp/sec) | "
                f"train loss: {trainer.loss.item():.5f} | "
                f"time remaining: {time_left:.2f} min"
            )
        logger.info(message)

    return log_training_progress

def train_model(
    config,
    model: torch.nn.Module,
    trainer: Trainer,
    dataset: torch.utils.data.Dataset,
    schedueler: torch.optim.lr_scheduler.LRScheduler,
    optimizer: torch.optim.Optimizer
) -> tuple:
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

    config.device = device

    log_fn = create_log_function(config)
    save_fn = create_save_function(config)

    trainer.run(
        config=config.trainer,
        model=model,
        train_dataset=dataset,
        scheduler=schedueler,
        optimizer=optimizer,
        log_fn=log_fn,
        save_fn=save_fn
    )

    logger.info("Training completed!")
    return model, trainer

def load_config_from_args(x: str):
    attempts = [
        x, 
        os.path.join(TRAIN_CONFIGS_PATH, x),
        os.path.join(PROJECT_ROOT, x)
    ]
    config_filepath = resolve(x, attempts)

    if config_filepath is None:
        raise FileNotFoundError(f"The train config files '{attempts}' does not exist.")
    if os.path.splitext(config_filepath)[1] != ".yaml":
        raise FileNotFoundError(f"{config_filepath} is not a .yaml")
        
    return config_filepath, load_config(config_filepath)

def load_config_from_run(run_dir: str):
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"The run dir '{run_dir}' does not exist.")
    if not os.path.isfile(os.path.join(run_dir, "train_config.yaml")):
        raise FileNotFoundError(f"Run dir '{run_dir}' does not contain \"train_config.yaml\"")
    
    return os.path.join(run_dir, "train_config.yaml"), load_config(os.path.join(run_dir, "train_config.yaml"))

def resolve_run_dir_from_args(x: str):
    attempts = [
        x,
        os.path.join(RUNS_PATH, x)
    ]
    run_dir = resolve(x, attempts, isfile=False)

    if run_dir is None:
        raise FileNotFoundError(f"The run dirs '{attempts}' does not exist.")
    
    return run_dir

def load_tokenizer_from_config(config) -> Tokenizer:
    dataset_dir = DATASET_DIRS[config.dataset]
    tokenizer_filepath = os.path.join(dataset_dir, "tokenizer.model")
    tokenizer = Tokenizer.load(tokenizer_filepath)
    return tokenizer

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train transformer model")
    parser.add_argument("--run_name", type=str, default=None, help="Name of the output directory")
    parser.add_argument("--config", type=str, default=None, help="Name of train config file")
    parser.add_argument("--run_dir", type=str, default=None, help="Name of run directory to restart training from")
    parser.add_argument("-o", action="store_true", help="If run name already taken, delete it and make a new one")
    args = parser.parse_args()

    global RUN_DIR

    if args.run_dir is None:
        if run_name_taken(args.run_name):
            if args.o:
                shutil.rmtree(os.path.join(RUNS_PATH, args.run_name))
            else:
                raise ValueError(f"The run name {args.run_name} is already taken")
            
        RUN_DIR = os.path.join(RUNS_PATH, args.run_name)
        os.makedirs(RUN_DIR, exist_ok=True)
        setup_logging(RUN_DIR)

        config_filepath, config = load_config_from_args(args.config)

        model = create_model(config)
        trainer = create_trainer(config)
        dataset = create_dataset(config)
        optimizer = create_optimizer(config, model)
        scheduler = create_scheduler(config, optimizer)
    else:
        RUN_DIR = resolve_run_dir_from_args(args.run_dir)
        setup_logging(RUN_DIR)

        if args.config is None:
            config_filepath, config = load_config_from_run(RUN_DIR)
        else:
            config_filepath, config = load_config_from_args(args.config)

    # copy config file and tokenizer model to run dir
    shutil.copy(config_filepath, os.path.join(RUN_DIR, "train_config.yaml"))
    shutil.copy(os.path.join(DATASET_DIRS[config.dataset], "tokenizer.model"), os.path.join(RUN_DIR, "tokenizer.model"))

    # Setup
    logger = logging.getLogger(__name__)
    logger.info("Starting training script")

    repo = Repo(search_parent_directories=True)
    git_hash = repo.head.object.hexsha
    logger.info(f"Git commit hash: {git_hash}")

    # Load configuration
    logger.info(f"Training configuration loaded")
    logger.info(f"Config:\n{pprint.pformat(namespace_to_dict(config))}")
    logger.info(f"Using device: {device}")

    try:
        train_model(config, model, trainer, dataset, scheduler, optimizer)
        logger.info("Training script completed successfully!")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
