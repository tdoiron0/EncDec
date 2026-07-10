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

        # Note: These could be a single batched linear layer
        # but we separate them for simplicity of implementation.
        self.k = nn.Linear(config.n_embd, config.n_embd)
        self.q = nn.Linear(config.n_embd, config.n_embd)
        self.v = nn.Linear(config.n_embd, config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.hidden_dropout = nn.Dropout(config.hidden_pdrop)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

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

        # get implicit dimension of each head
        head_dim = int(C / self.n_head)

        # get intermediate matrices from applying weight matrices and split along embedding dimension to get heads
        Q = self.q(x).view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, T, C) @ (C, C) = (B, T, C)  →  (B, T, H, d)  →  (B, H, T, d)
        K = self.k(x).view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, T, C) @ (C, C) = (B, T, C)  →  (B, T, H, d)  →  (B, H, T, d)
        V = self.v(x).view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, T, C) @ (C, C) = (B, T, C)  →  (B, T, H, d)  →  (B, H, T, d)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(head_dim) # (B, H, T, d) @ (B, H, d, T) = (B, H, T, T)

        # apply attention mask to attention scores
        scores = scores.masked_fill(attention_mask == 0, float("-inf")) # (B, H, T, T)

        # obtain attention weights
        attn_weights = F.softmax(scores, dim=-1) # (B, H, T, T)

        # dropout regularization
        attn_weights = self.attn_dropout(attn_weights) # (B, H, T, T)

        # obtain hidden states for each head
        y = attn_weights @ V # (B, H, T, T) @ (B, H, T, d) = (B, H, T, d)

        # concat and final linear projection
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, H, T, d)  →  (B, T, H, d)  →  (B, T, C)
        y = self.c_proj(y) # (B, T, C) @ (C, C) = (B, T, C)

        # more dropout regularization
        y = self.hidden_dropout(y) # (B, T, C)

        return y