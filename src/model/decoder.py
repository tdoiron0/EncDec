import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from jaxtyping import Float, Int 

from src.model.embedding import Embedding
from src.model.decoder_block import DecoderBlock

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.decoder = nn.ModuleDict(
            dict(
                embedding=Embedding(config),
                h=nn.ModuleList(
                    [DecoderBlock(config) for _ in range(config.n_layer)]
                )
            )
        )
    
    def forward(
        self,
        idx: Int[torch.Tensor, "batch tgt_seq_len"],
        c: Float[torch.Tensor, "batch src_seq_len n_embd"], 
        generic_attn_mask: Int[torch.Tensor, "batch 1 tgt_seq_len src_seq_len"],
        cross_attn_mask: Int[torch.Tensor, "batch 1 tgt_seq_len src_seq_len"],
    ) -> Float[torch.Tensor, "batch seq_len n_embd"]:
        x = self.decoder.embedding(idx)
        for block in self.decoder.h:
            x = block(x, c, generic_attn_mask, cross_attn_mask)
        return x