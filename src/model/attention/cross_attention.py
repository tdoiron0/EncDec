import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from jaxtyping import Float, Int 

class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # tensor dimension restriction specific to this implementation
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.q_proj = nn.Linear(self.n_embd, self.n_embd)
        self.kv_proj = nn.Linear(self.n_embd, 2 * self.n_embd)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd)

        self.dropout_p = config.attn_pdrop

    def forward(
        self,
        x: Float[torch.Tensor, "batch tgt_seq_len n_embd"],
        h: Float[torch.Tensor, "batch src_seq_len n_embd"], 
        attention_mask: Int[torch.Tensor, "batch 1 tgt_seq_len src_seq_len"]
    ) -> Float[torch.Tensor, "batch tgt_seq_len n_embd"]:
        """
        Implement multi-headed self-attention in GPT-2 Style.
        """
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = (x.size())
        B, U, C = (h.size())

        head_dim = int(C / self.n_head)

        q = self.q_proj(x)
        kv = self.kv_proj(h)
        k, v = kv.chunk(2, dim=-1)

        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, U, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, U, self.n_head, head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            # SDPA bool mask semantics: True = attend, so valid (nonzero) positions
            attn_mask=(attention_mask != 0),
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )

        y = attn.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)

        return y