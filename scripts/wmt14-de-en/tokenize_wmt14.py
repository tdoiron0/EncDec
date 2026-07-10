import os
import argparse
import yaml
import torch
import shutil
from pprint import pprint

from constants.constants import WMT14_PROC_PATH, WMT14_RAW_DIR, PREPROCESS_CONFIG_PATH, TOKENIZERS_PATH
from src.tokenizer.tokenizer import Tokenizer
from utils import load_config

def get_tokenizer_filepath(config):
    return os.path.join(TOKENIZERS_PATH, f"{config.tokenizer}.model")

def tokenize_dataset(config) -> int:
    tokenizer = Tokenizer.load(get_tokenizer_filepath(config))
    bos, eos = tokenizer.bos_id(), tokenizer.eos_id()
    chunk_size = 100000
    block_size = 0
    dropped = 0

    def flush(en_batch, de_batch, src_flat, tgt_flat, src_offsets, tgt_offsets):
        nonlocal block_size, dropped
        src_ids = tokenizer.encode_batch(de_batch)
        tgt_ids = tokenizer.encode_batch(en_batch)
        for s, t in zip(src_ids, tgt_ids):
            src = [bos] + s + [eos]
            tgt = [bos] + t + [eos]
            if max(len(src), len(tgt)) > config.max_seq_len:
                dropped += 1
                continue
            block_size = max(block_size, len(src), len(tgt))
            src_flat.extend(src)
            tgt_flat.extend(tgt)
            src_offsets.append(src_offsets[-1] + len(src))
            tgt_offsets.append(tgt_offsets[-1] + len(tgt))

    for split in ("train", "validation", "test"):
        print(f"Tokenizing {split}:")
        dropped = 0
        src_flat = []
        tgt_flat = []
        src_offsets = [0]
        tgt_offsets = [0]
        en_batch = []
        de_batch = []
        with open(os.path.join(WMT14_RAW_DIR, split), encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                cols = line.rstrip("\n").split("\t")
                if len(cols) == 2:
                    en_batch.append(cols[0])
                    de_batch.append(cols[1])
                if len(en_batch) >= chunk_size:
                    flush(en_batch, de_batch, src_flat, tgt_flat, src_offsets, tgt_offsets)
                    en_batch.clear()
                    de_batch.clear()
                    print(f"Processed {i} lines")
        if en_batch:
            flush(en_batch, de_batch, src_flat, tgt_flat, src_offsets, tgt_offsets)
        kept = len(src_offsets) - 1
        print(f"Dropped {dropped} pairs longer than {config.max_seq_len} tokens ({kept} kept)")
        print(f"Saving {split} to file")

        pairs = {
            "src": torch.tensor(src_flat, dtype=torch.int16),
            "tgt": torch.tensor(tgt_flat, dtype=torch.int16),
            "src_offsets": torch.tensor(src_offsets, dtype=torch.int64),
            "tgt_offsets": torch.tensor(tgt_offsets, dtype=torch.int64)
        }
        torch.save(pairs, os.path.join(WMT14_PROC_PATH, f"{split}.pt"))

    return block_size

def main():
    parser = argparse.ArgumentParser(description="Tokenize WMT14 de-en dataset")
    parser.add_argument(
        "--config", type=str, default="wmt14-de-en1.yaml", help="Config file path"
    )
    args = parser.parse_args()

    config_filepath = os.path.join(PREPROCESS_CONFIG_PATH, args.config)
    config = load_config(config_filepath)
    print(f"Loaded config file: {args.config}")
    pprint(config)
    
    if os.path.isdir(WMT14_PROC_PATH):
        shutil.rmtree(WMT14_PROC_PATH)
    os.makedirs(WMT14_PROC_PATH, exist_ok=True)

    block_size = tokenize_dataset(config)
    print(f"block_size (max sequence length across all splits): {block_size}")

    # Add block_size to config and save in new processed dataset directory
    with open(config_filepath, "r") as f:
        config_raw = yaml.safe_load(f)
    config_raw["block_size"] = block_size
    with open(os.path.join(WMT14_PROC_PATH, "config.yaml"), "w") as f:
        yaml.dump(config_raw, f, allow_unicode=True)
    
    # Copy the tokenizer model to processed dataset directory
    shutil.copy(get_tokenizer_filepath(config), os.path.join(WMT14_PROC_PATH, "tokenizer.model"))

if __name__ == "__main__":
    main()
