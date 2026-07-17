import os
import torch
from torch.optim.lr_scheduler import LRScheduler, LinearLR, CosineAnnealingLR, SequentialLR
from torch.optim.optimizer import Optimizer
from typing import Any, Callable

from src.trainer.trainer import Trainer
from src.model.enc_dec import EncoderDecoder
from src.datasets.JESC_dataset import JESCDataset
from src.datasets.wmt14_dataset import WMT14Dataset

import constants as c

MODEL_INDEX: dict[str, torch.nn.Module] = {
    c.ENC_DEC_NAME: EncoderDecoder
}

TRAINER_INDEX: dict[str, Trainer] = {
    c.DEFAULT_TRAINER_NAME: Trainer
}

DATASET_INDEX: dict[str, torch.utils.data.Dataset] = {
    c.JESC_TOK_NAME: JESCDataset,
    c.WMT14_TOK_NAME: WMT14Dataset
}

OPTIMIZER_INDEX: dict[str, Optimizer] = {
    c.ADAMW_NAME: torch.optim.AdamW
}

def init_sequential_lr(config, optimizer) -> SequentialLR:
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=config.scheduler.start_factor, 
        end_factor=config.scheduler.end_factor, 
        total_iters=config.scheduler.warmup_epochs
    )
    
    main_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=(config.trainer.max_epochs - config.scheduler.warmup_epochs)
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[config.scheduler.warmup_epochs]
    )

SCHEDULER_INDEX: dict[str, Callable[[Any, Optimizer], LRScheduler]] = {
    c.SEQUENTIAL_LR_SCHEDULER_NAME: init_sequential_lr
}

DTYPE_INDEX: dict[str, torch.dtype] = {
    c.FLOAT16: torch.float16,
    c.BFLOAT16: torch.bfloat16,
}