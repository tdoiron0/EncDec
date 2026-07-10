import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from jaxtyping import Float, Int 

"""
Implemtation of an absolute learned positional embedder
"""
class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embeddings = nn.Embedding(config.block_size, config.n_embd)
    
    def forward(
        self, idx: Int[torch.Tensor, "batch seq_len"]
    ) -> Float[torch.Tensor, "batch seq_len n_embd"]:
        B, T = idx.size()

        token_embeddings = self.vocab_embeddings(idx)
        positions = torch.arange(T, device=idx.device)
        pos_embeddings = self.position_embeddings(positions).unsqueeze(0)

        embeddings = token_embeddings + pos_embeddings

        return embeddings