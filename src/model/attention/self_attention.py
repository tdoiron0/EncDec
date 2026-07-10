import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from jaxtyping import Float, Int 

@dataclass
class SelfAttentionConfig:
    D_x: int = 2
    D_q: int = 2
    D_v: int = 2

class SelfAttention(nn.Module):
    def __init__(self, config: SelfAttentionConfig):
        super().__init__()

        self.D_q = config.D_q

        self.W_q = nn.Parameter(torch.empty(config.D_x, config.D_q))
        self.W_k = nn.Parameter(torch.empty(config.D_x, config.D_q))
        self.W_v = nn.Parameter(torch.empty(config.D_x, config.D_v))

        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Q = X @ self.W_q                          # (N_x, D_q)
        K = X @ self.W_k                          # (N_x, D_q)
        V = X @ self.W_v                          # (N_x, D_v)
        E = (Q @ K.transpose(-2, -1)) / math.sqrt(self.D_q)  # (N_q, N_x)
        A = F.softmax(E, dim=-1)                  # row-wise softmax
        Y = A @ V                                 # (N_q, D_v)
        return Y