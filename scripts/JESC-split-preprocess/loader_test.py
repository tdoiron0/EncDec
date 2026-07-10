import torch 
import numpy as np 
from torch.utils.data import DataLoader
import os
import sys

from src.datasets.JESC_dataset import JESCDataset
from src.datasets.bucket_sampler import pad_collate
from constants.constants import PAD_TOKEN

BATCH_SIZE = 16
NUM_WORKERS = 0
device = "cpu"

dataset = JESCDataset("test", 107)
loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    collate_fn=pad_collate,
)

# print(dataset.__getitem__(1))

# for batch in loader:
#     src, tgt = batch
#     for it in tgt:
#         if 3 in it:
#             print(it)

src, tgt = next(iter(loader))

src_count = (src != PAD_TOKEN).type(torch.int).sum(dim=1)
tgt_count = (tgt != PAD_TOKEN).type(torch.int).sum(dim=1)
src = src[:, : src_count.max().item()]
tgt = tgt[:, : tgt_count.max().item()]

print(src[0])
print(tgt[0])