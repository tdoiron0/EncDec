import torch 
import numpy as np 
from torch.utils.data import DataLoader
import os
import sys

from src.datasets.JESC_dataset import JESCDataset
from src.datasets.bucket_sampler import pad_collate
from src.model.enc_dec import EncoderDecoder
from config.train_config import TrainConfig

np.set_printoptions(precision=1)
np.set_printoptions(linewidth=200)

BATCH_SIZE = 16
NUM_WORKERS = 0
DEVICE = "cpu"

config=TrainConfig(
    vocab_size=32000,
    block_size=109,
    n_layer=2,
    n_embd=16,
    n_head=4,
    hidden_pdrop=0.0,
    attn_pdrop=0.0,
)
model = EncoderDecoder(config)

dataset = JESCDataset("test", 107)
loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    collate_fn=pad_collate,
)

src, tgt = next(iter(loader))

logits, loss = model(src, tgt)

print(next(model.parameters()).dtype)

print(logits)
print(loss)