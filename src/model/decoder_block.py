import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from jaxtyping import Float, Int 

from src.model.attention.generic_self_attention import GenericSelfAttention
from src.model.attention.cross_attention import CrossAttention
from src.model.new_GELU import NewGELU

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn1 = GenericSelfAttention(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn2 = CrossAttention(config)
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
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

        self.ln_3 = nn.LayerNorm(config.n_embd)
    
    def forward(
        self,
        x: Float[torch.Tensor, "batch tgt_seq_len n_embd"],
        c: Float[torch.Tensor, "batch src_seq_len n_embd"],
        generic_attn_mask: Int[torch.Tensor, "batch 1 tgt_seq_len src_seq_len"],
        cross_attn_mask: Int[torch.Tensor, "batch 1 tgt_seq_len src_seq_len"],
    ) -> Float[torch.Tensor, "batch seq_len n_embd"]:
        x = self.ln_1(x + self.attn1(x, generic_attn_mask))
        x = self.ln_2(x + self.attn2(x, c, cross_attn_mask))
        x = self.ln_3(x + self.mlpf(x))
        return x