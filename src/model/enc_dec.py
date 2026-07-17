import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from jaxtyping import Float, Int 

from src.model.encoder import Encoder
from src.model.decoder import Decoder
from constants import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN

class EncoderDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.block_size = config.block_size

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.m = nn.Softmax()

        self.apply(self._init_weights)

        # scaled init for the residual projections (GPT-2 §2.3)
        for pn, p in self.named_parameters():
            if pn.endswith(("c_proj.weight", "out_proj.weight")):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def configure_optimizers(self, weight_decay: float) -> list:
        """Split parameters into weight-decay / no-decay groups (GPT-2 convention).

        Matmul weights (nn.Linear) are decayed; biases, LayerNorm gains/biases, and
        embedding tables are not. Iterating with ``recurse=False`` visits each
        parameter exactly once at its owning leaf module, so every parameter is
        classified without the brittle set-completeness assertions minGPT uses.
        """
        decay, no_decay = set(), set()
        for module_name, module in self.named_modules():
            for param_name, _ in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                if param_name.endswith("bias"):
                    no_decay.add(full_name)
                elif isinstance(module, (nn.LayerNorm, nn.Embedding)):
                    no_decay.add(full_name)
                else:
                    decay.add(full_name)

        param_dict = dict(self.named_parameters())
        return [
            {"params": [param_dict[n] for n in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[n] for n in sorted(no_decay)], "weight_decay": 0.0},
        ]

    def get_attention_mask(
        self, num_tokens: Int[torch.Tensor, "batch"]
    ) -> Int[torch.Tensor, "batch 1 max_tokens max_tokens"]:
        """
        Decoder self-attention mask: a causal (lower-triangular) mask combined
        with a padding mask. Query position i may attend to key position j iff
        ``j <= i`` (causality) and ``j < num_tokens[b]`` (not padding).

        :param num_tokens: number of valid tokens per batch element, shape (batch,)
        :returns attention_mask: mask of shape (batch, 1, max_tokens, max_tokens)
        """
        B = num_tokens.shape[0]
        max_tokens = min(self.block_size, num_tokens.max().item())

        # padding mask: key position j is valid when j < num_tokens[b]  -> (B, max_tokens)
        pad_mask = (
            torch.arange(max_tokens, device=num_tokens.device) < num_tokens.unsqueeze(1)
        ).to(torch.int)
        pad_mask = pad_mask[:, None, None, :].expand(B, 1, max_tokens, max_tokens)

        # causal mask: zero out keys j > i for every query row i
        return torch.tril(pad_mask)

    def forward(
        self,
        src_idx: Int[torch.Tensor, "batch src_seq_len"],
        tgt_idx: Int[torch.Tensor, "batch tgt_seq_len"]
    ) -> Float[torch.Tensor, "batch tgt_seq_len vocab_size"]:
        src_count = (src_idx != PAD_TOKEN).type(torch.int).sum(dim=1)   # Int[batch]
        tgt_count = (tgt_idx != PAD_TOKEN).type(torch.int).sum(dim=1)   # Int[batch]

        src_idx = src_idx[:, : src_count.max().item()]  # 
        tgt_idx = tgt_idx[:, : tgt_count.max().item()]

        # teacher forcing: input is the target shifted right, labels shifted left.
        # clone so the in-place padding below does not mutate tgt_idx (and tgt_out_idx).
        tgt_in_idx = tgt_idx[:, :-1].clone()
        for i, end in enumerate(tgt_count):
            tgt_in_idx[i, end - 1:] = PAD_TOKEN
        tgt_out_idx = tgt_idx[:, 1:]

        # encoder self-attention: mask padded source keys -> (B, 1, 1, src_seq_len)
        src_pad_mask = (src_idx != PAD_TOKEN)[:, None, None, :]
        c = self.encoder(src_idx, src_pad_mask)

        # decoder self-attention: causal + target-padding mask. One fewer token than
        # tgt_count because the last target token is dropped from the decoder input.
        dec_self_mask = self.get_attention_mask(tgt_count - 1)

        # decoder cross-attention: queries are target, keys/values are the source
        # encoding -> reuse the source padding mask (broadcasts over target queries).
        z = self.decoder(tgt_in_idx, c, dec_self_mask, src_pad_mask)
        logits = self.lm_head(z)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out_idx.reshape(-1),
            ignore_index=PAD_TOKEN,
            label_smoothing=0.1
        )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        src_idx: Int[torch.Tensor, "1 src_seq_len"],
        max_new_tokens: int,
        bos_id: int = BOS_TOKEN,
        eos_id: int = EOS_TOKEN,
        temperature: float = 0.0,
        beam_width: int = 5,
        length_penalty: float = 0.6,
    ) -> list:
        """Translate a single source sequence with beam search.

        ``src_idx`` is a ``(1, src_seq_len)`` tensor of source ids (with BOS/EOS,
        optionally PAD-padded). The source is encoded once and the encoding is
        shared across beams; each step re-runs the decoder over every beam's full
        prefix (no KV cache). All beams start from BOS and grow in lockstep; a beam
        that emits ``eos_id`` is frozen (extended with PAD at zero cost) so its
        score stays comparable while the rest keep searching. Search ends when all
        beams have finished, after ``max_new_tokens`` steps, or when the target
        fills the model's block size.

        ``beam_width == 1`` is greedy decoding. ``temperature > 0`` softens the
        logits before scoring (selection stays deterministic top-k). The final
        hypothesis is chosen by GNMT length-normalized score,
        ``log_prob / ((5 + len) / 6) ** length_penalty``.

        Returns the generated target ids as a python list, excluding the leading BOS
        and the terminating EOS.
        """
        was_training = self.training
        self.eval()
        device = src_idx.device

        # Encode the source once: trim trailing pad, build the source padding mask.
        src_count = (src_idx != PAD_TOKEN).type(torch.int).sum(dim=1)
        src_idx = src_idx[:, : src_count.max().item()]
        src_pad_mask = (src_idx != PAD_TOKEN)[:, None, None, :]
        c = self.encoder(src_idx, src_pad_mask)

        # Share the source encoding across beams (expand: no copy, decoder only reads).
        c = c.expand(beam_width, -1, -1)
        src_pad_mask = src_pad_mask.expand(beam_width, -1, -1, -1)

        # All beams start identical, so only beam 0 gets a live score; the rest sit
        # at -inf so the first top-k draws all candidates from beam 0's row.
        beams = torch.full((beam_width, 1), bos_id, dtype=torch.long, device=device)
        scores = torch.full((beam_width,), float("-inf"), device=device)
        scores[0] = 0.0
        finished = torch.zeros(beam_width, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            if finished.all() or beams.size(1) >= self.block_size:
                break

            t = beams.size(1)
            self_mask = self.get_attention_mask(
                torch.full((beam_width,), t, dtype=torch.long, device=device)
            )
            out = self.decoder(beams, c, self_mask, src_pad_mask)
            logits = self.lm_head(out[:, -1, :])
            if temperature > 0:
                logits = logits / temperature
            log_probs = F.log_softmax(logits, dim=-1)

            # Finished beams may only extend with PAD at zero cost, freezing them.
            log_probs[finished] = float("-inf")
            log_probs[finished, PAD_TOKEN] = 0.0

            total = scores.unsqueeze(1) + log_probs
            scores, top_idx = total.view(-1).topk(beam_width)
            beam_src = top_idx // log_probs.size(1)
            next_tok = top_idx % log_probs.size(1)

            beams = torch.cat([beams[beam_src], next_tok.unsqueeze(1)], dim=1)
            finished = finished[beam_src] | (next_tok == eos_id)

        # Pick the best hypothesis by length-normalized score (GNMT §7), where a
        # hypothesis's length counts generated tokens up to and including EOS.
        gen = beams[:, 1:]
        lengths = torch.full((beam_width,), gen.size(1), dtype=torch.float, device=device)
        for i in range(beam_width):
            eos_pos = (gen[i] == eos_id).nonzero()
            if eos_pos.numel() > 0:
                lengths[i] = eos_pos[0].item() + 1
        norm = ((5.0 + lengths) / 6.0) ** length_penalty
        best = (scores / norm).argmax().item()

        if was_training:
            self.train()

        ids = gen[best].tolist()
        if eos_id in ids:
            ids = ids[: ids.index(eos_id)]
        return ids
    
    def load(self, checkpoint):
        return super().load_state_dict(checkpoint['model_state_dict'])