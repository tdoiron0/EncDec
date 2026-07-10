import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from jaxtyping import Float, Int 

from src.model.embedding import Embedding
from src.model.encoder_block import EncoderBlock

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = nn.ModuleDict(
            dict(
                embedding=Embedding(config),
                h=nn.ModuleList(
                    [EncoderBlock(config) for _ in range(config.n_layer)]
                )
            )
        )

    def forward(
        self,
        idx: Int[torch.Tensor, "batch src_seq_len"],
        pad_mask: Int[torch.Tensor, "batch 1 src_seq_len src_seq_len"]
    ) -> Float[torch.Tensor, "batch src_seq_len n_embd"]: 
        x = self.encoder.embedding(idx)
        for block in self.encoder.h:
            x = block(x, pad_mask)
        return x
        