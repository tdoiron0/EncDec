import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from jaxtyping import Float, Int 

from src.model.transformer_block import TransformerBlock
from src.model.embedding import Embedding

@dataclass
class TransformerConfig:
    """Configuration for transformer models."""

    vocab_size: int = 10
    block_size: int = 10
    n_layer: int = 2
    n_embd: int = 4
    n_head: int = 2
    hidden_pdrop: float = 0.1
    attn_pdrop: float = 0.1 

class GenericTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        self.transformer = nn.ModuleDict(
            dict(
                embedding=Embedding(config),
                h=nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.n_layer)]
                ),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters (note we only count transformer parameters, not lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        # print("number of parameters: %.2fM" % (n_params / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the optimizer parameter groups for PyTorch optimizer initialization.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, (
            "parameters %s made it into both decay/no_decay sets!"
            % (str(inter_params),)
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(param_dict.keys() - union_params),)
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        return optim_groups

    def get_attention_mask(
        self, num_tokens: Int[torch.Tensor, "batch"]
    ) -> Int[torch.Tensor, "batch 1 max_tokens max_tokens"]:
        """
        Base implementation - subclasses will override this for specific attention patterns.

        :param num_tokens: Number of tokens per batch element of shape (batch,)
        :returns attention_mask: Attention mask of shape (batch, 1, max_tokens, max_tokens)
        """
        B = num_tokens.shape[0]
        max_tokens = min(self.block_size, num_tokens.max().item())
        return torch.ones((B, 1, max_tokens, max_tokens), dtype=torch.int)

    def forward(
        self,
        idx: Int[torch.Tensor, "batch seq_len"],
        targets: Optional[Int[torch.Tensor, "batch seq_len"]] = None,
        return_hidden: bool = False,
    ) -> Tuple[
        Float[torch.Tensor, "batch seq_len vocab_size"],
        Optional[Float[torch.Tensor, ""]],
    ]:
        """
        Put all the modules of a Transformer together for inference

        All the modules you'll need are defined in self.transformer
        - You can iterate through a nn.ModuleList using a standard for loop.
        - Make sure to apply layer normalization (ln_f) after the final transformer block and before the language modeling head.
        - the hidden state is the output of the final transformer block after layer normalization but before the language modeling head

        This will take a few lines!

        :param idx: Token indices of shape (batch, seq_len)
        :param targets: Target token indices of shape (batch, seq_len), optional
        :param return_hidden: Whether to return the hidden state
        :returns logits: Output logits of shape (batch, seq_len, vocab_size)
        :returns loss: Cross-entropy loss (scalar) or None if targets not provided
        :returns hidden: Hidden state of shape (batch, seq_len, n_embd) if return_hidden is True, otherwise None
        """

        num_tokens = (idx != -1).type(torch.int).sum(dim=1)
        idx = idx.masked_fill(idx == -1, 0).type(torch.int)[
            :, : num_tokens.max().item()
        ]
        if targets is not None:
            targets = targets[:, : num_tokens.max().item()]
        attention_mask = self.get_attention_mask(num_tokens)

        ### TODO: BEGIN SOLUTION ###
        
        attention_mask = attention_mask.to(idx.device)
        x = self.transformer.embedding(idx)
        for block in self.transformer.h:
            x = block(x, attention_mask)
        hidden = self.transformer.ln_f(x)
        logits = self.lm_head(hidden)

        ### END SOLUTION ###

        if return_hidden:
            return hidden
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-1,
            )

        return logits, loss