import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from jaxtyping import Float, Int 

'''
This is an implementation of the masked, multi-headed, self-attention layer 
(generic self-attention)
'''
class GenericSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # tensor dimension restriction specific to this implementation
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.qkv_proj = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.dropout_p = config.attn_pdrop

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq_len n_embd"],
        attention_mask: Int[torch.Tensor, "batch 1 seq_len seq_len"],
    ) -> Float[torch.Tensor, "batch seq_len n_embd"]:
        """
        Implement multi-headed self-attention in GPT-2 Style.
        """
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = (x.size())
        head_dim = int(C / self.n_head)

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

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