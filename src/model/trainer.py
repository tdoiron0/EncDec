import time
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Callable

import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from src.datasets.bucket_sampler import BucketBatchSampler, pad_collate

from utils import get_best_device
from constants.constants import PAD_TOKEN

class Trainer:
    def __init__(
        self,
        config,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        log_fn: Optional[Callable] = None,
        save_fn: Optional[Callable] = None,
    ):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.log_fn = log_fn  # called every log_interval iters
        self.save_fn = save_fn
        self.logger = logging.getLogger(__name__)
        self.rate_lr = 0.001

        # determine the device we'll train on
        if config.device == "auto":
            self.device = get_best_device()
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        # print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.epoch = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.loss = None

    def run(self):
        model, config = self.model, self.config

        self.optimizer = torch.optim.AdamW(
            model.configure_optimizers(config.weight_decay),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
        )

        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=config.warmup_iters)
        main_scheduler = CosineAnnealingLR(self.optimizer, T_max=(config.max_epochs*self.train_dataset.__len__() - config.warmup_iters))
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler], 
            milestones=[config.warmup_iters]
        )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True
        )
        
        use_amp = self.device == "cuda"

        amp_dtype = torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

        if getattr(config, "compile", False) and self.device == "cuda":
            model = torch.compile(model)

        self.iter_time = time.time()
        self.total_samps = self.train_dataset.__len__()
        self.rate = 0
        self.pad_waste = 0

        for epoch in range(config.max_epochs):
            self.epoch = epoch
            self.samps = 0
            self.iter_num = 0
            model.train()

            for batch in train_loader:
                self.optimizer.zero_grad(set_to_none=True)  

                batch = [t.to(self.device) for t in batch]
                src, tgt = batch

                # in the batch loop:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    logits, self.loss = model(src, tgt)

                self.scaler.scale(self.loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                self.iter_num += 1
                tnow = time.time()
                self.iter_dt = tnow - self.iter_time
                self.iter_time = tnow

                self.iter_samps = src.shape[0]
                self.samps += self.iter_samps

                self.rate = self.rate*(1-self.rate_lr) + (self.iter_samps / self.iter_dt)*self.rate_lr

                if self.log_fn and self.iter_num % config.log_interval == 0:
                    self.log_fn(self)
        
            self.save_fn(self)