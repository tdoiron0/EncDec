import math

import torch
from torch import nn
from jaxtyping import Float, Int

"""
Implementation of a fixed sinusoidal positional embedder (Vaswani et al. 2017, §3.5)
"""
class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_embd = config.n_embd

        self.vocab_embeddings = nn.Embedding(config.vocab_size, config.n_embd)

        # frequencies 1 / 10000^(2i/d); the position axis is built per forward
        # pass, so there is no fixed maximum sequence length
        div_term = torch.exp(torch.arange(0, config.n_embd, 2) * (-math.log(10000.0) / config.n_embd))
        self.register_buffer("div_term", div_term, persistent=False)

    def forward(
        self, idx: Int[torch.Tensor, "batch seq_len"]
    ) -> Float[torch.Tensor, "batch seq_len n_embd"]:
        B, T = idx.size()

        # scale tokens by sqrt(n_embd) so they aren't drowned out by the
        # unit-amplitude sinusoids (Vaswani et al. 2017, §3.4)
        token_embeddings = self.vocab_embeddings(idx) * math.sqrt(self.n_embd)

        # PE[pos, 2i] = sin(pos / 10000^(2i/d)), PE[pos, 2i+1] = cos(...)
        pos = torch.arange(T, device=idx.device).unsqueeze(1)
        pe = torch.empty(T, self.n_embd, device=idx.device, dtype=token_embeddings.dtype)
        pe[:, 0::2] = torch.sin(pos * self.div_term)
        pe[:, 1::2] = torch.cos(pos * self.div_term)

        return token_embeddings + pe
