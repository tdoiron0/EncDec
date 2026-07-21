import time
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Callable

import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from utils import get_best_device
from constants import DTYPE_INDEX
from src.datasets.bucket_sampler import pad_collate

class Trainer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = get_best_device()
        self.rate_lr = 0.0001
        
    def run(
        self,
        config,
        model: torch.nn.Module, 
        train_dataset: torch.utils.data.Dataset,
        scheduler: torch.optim.lr_scheduler.LRScheduler, 
        optimizer: torch.optim.Optimizer,
        log_fn,
        save_fn
    ):
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            collate_fn=pad_collate
        )
        
        use_amp = self.device == "cuda"
        amp_dtype = DTYPE_INDEX[config.amp_dtype]
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

        if getattr(config, "compile", False) and self.device == "cuda":
            model = torch.compile(model)

        self.iter_time = time.time()
        self.total_samps = train_dataset.__len__()
        self.rate = 0
        self.pad_waste = 0

        for epoch in range(config.max_epochs):
            self.epoch = epoch
            self.samps = 0
            self.iter_num = 0
            model.train()

            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)  

                batch = [t.to(self.device) for t in batch]
                src, tgt = batch

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    logits, self.loss = model(src, tgt)

                self.scaler.scale(self.loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.scaler.step(optimizer)
                self.scaler.update()

                self.iter_num += 1
                tnow = time.time()
                self.iter_dt = tnow - self.iter_time
                self.iter_time = tnow

                self.iter_samps = src.shape[0]
                self.samps += self.iter_samps

                self.rate = self.rate*(1-self.rate_lr) + (self.iter_samps / self.iter_dt)*self.rate_lr

                if log_fn and self.iter_num % config.log_interval == 0:
                    log_fn(self)

            scheduler.step()
            save_fn(self)