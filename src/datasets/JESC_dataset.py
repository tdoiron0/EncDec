from torch.utils.data import Dataset
import torch
import sys, os

from constants import JESC_TOK_DIR

class JESCDataset(Dataset):
    """Prefix-LM dataset for Japanese -> English translation.

    Consumes the pre-tokenized splits written by
    scripts/JESC-split-preprocess/JESC_tokenize.py: each split is a
    torch.save'd list of (src_ids, tgt_ids), where
        src = [BOS] + encode(ja) + [EOS]   (EOS doubles as separator)
        tgt = encode(en) + [EOS]
    Each item is the concatenation src + tgt framed as next-token prediction.
    Loss is supervised on the English half only; source positions in y are -1
    (the model's ignore_index), so the Japanese prefix is read but not predicted.
    """

    DATA_DIR = JESC_TOK_DIR

    def __init__(self, split: str):
        assert split in {"train", "val", "test"}
        self.split = split
        self.pairs: dict[str, torch.Tensor] = torch.load(os.path.join(self.DATA_DIR, f"{split}.pt"))

    def __len__(self):
        return len(self.pairs.get("src_offsets")) - 1

    def __getitem__(self, i):
        src_start = self.pairs.get("src_offsets")[i]
        src_end = self.pairs.get("src_offsets")[i+1]
        tgt_start = self.pairs.get("tgt_offsets")[i]
        tgt_end = self.pairs.get("tgt_offsets")[i+1]

        src = self.pairs.get("src")[src_start:src_end]
        tgt = self.pairs.get("tgt")[tgt_start:tgt_end]

        # variable length; the DataLoader's pad_collate pads to the batch max
        return src.long(), tgt.long()