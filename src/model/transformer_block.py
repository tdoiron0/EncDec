import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from jaxtyping import Float, Int 

from src.model.attention.generic_self_attention import GenericSelfAttention
from src.model.new_GELU import NewGELU

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = GenericSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
                c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(config.hidden_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq_len n_embd"],
        attention_mask: Int[torch.Tensor, "batch 1 seq_len seq_len"],
    ) -> Float[torch.Tensor, "batch seq_len n_embd"]:
        x = x + self.attn(self.ln_1(x), attention_mask) # (B, T, C) + (B, T, C) = (B, T, C)
        x = x + self.mlpf(self.ln_2(x)) # 

        return x