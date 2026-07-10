import os
import torch

from constants.constants import JESC_PROC_PATH

SPLIT = "test.pt"

pairs: dict[str, torch.Tensor] = torch.load(os.path.join(JESC_PROC_PATH, SPLIT))
for i in range(len(pairs.get("tgt_offsets")) - 1):
    start = pairs.get("tgt_offsets")[i]
    end = pairs.get("tgt_offsets")[i+1]

    print(pairs.get("tgt")[start:end].numpy())