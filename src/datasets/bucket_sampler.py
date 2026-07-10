import random

import torch
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence

from constants.constants import PAD_TOKEN


class BucketBatchSampler(Sampler):
    """Yields batches of indices grouped by similar sequence length.

    Each epoch: shuffle all indices, split them into pools of
    ``batch_size * pool_factor``, sort within each pool by length, slice the
    pool into batches, and shuffle the batch order. Batch contents stay
    effectively random (drawn from a random pool, served in random order)
    while samples within a batch have near-identical lengths, so dynamic
    padding wastes almost nothing.
    """

    def __init__(self, lengths: torch.Tensor, batch_size: int, pool_factor: int = 100):
        self.lengths = lengths
        self.batch_size = batch_size
        self.pool_size = batch_size * pool_factor

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        perm = torch.randperm(len(self.lengths))
        for pool in perm.split(self.pool_size):
            pool = pool[torch.argsort(self.lengths[pool])]
            batches = list(pool.split(self.batch_size))
            random.shuffle(batches)
            for batch in batches:
                yield batch.tolist()


def pad_collate(batch):
    """Pad (src, tgt) pairs to the longest sequence in the batch.

    Replaces the default collate so batches are ``(B, batch_max_len)`` instead
    of ``(B, block_size)``; the model already trims/masks by PAD_TOKEN.
    """
    srcs, tgts = zip(*batch)
    src = pad_sequence(srcs, batch_first=True, padding_value=PAD_TOKEN)
    tgt = pad_sequence(tgts, batch_first=True, padding_value=PAD_TOKEN)
    return src, tgt
