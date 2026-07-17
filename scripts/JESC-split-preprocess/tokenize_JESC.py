import os
import argparse
import yaml
import torch
from types import SimpleNamespace

from constants import PROJECT_ROOT, JESC_TOK_DIR, JESC_RAW_DIR, PREPROCESS_CONFIG_PATH
from src.tokenizer.tokenizer import Tokenizer

def load_config(filename):
    def to_namespace(data):
        if isinstance(data, dict):
            return SimpleNamespace(
                **{key: to_namespace(value) for key, value in data.items()}
            )
        elif isinstance(data, list):
            return [to_namespace(item) for item in data]
        return data
    with open(os.path.join(PREPROCESS_CONFIG_PATH, filename), "r") as f:
        data = yaml.safe_load(f)
    return to_namespace(data)

def tokenize_dataset(config) -> int:
    tokenizer = Tokenizer.load(config.tokenizer)
    block_size = 0

    for split in ("train", "val", "test"):
        print(f"Tokenizing {split}:")
        src_flat = []
        tgt_flat = []
        src_offsets = [0]
        tgt_offsets = [0]
        with open(os.path.join(JESC_RAW_DIR, split), encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                cols = line.rstrip("\n").split("\t")
                if len(cols) == 2:
                    en, ja = cols

                    src = [tokenizer.bos_id()] + tokenizer.encode(ja) + [tokenizer.eos_id()]
                    tgt = [tokenizer.bos_id()] + tokenizer.encode(en) + [tokenizer.eos_id()]

                    block_size = max(block_size, len(src), len(tgt))

                    src_flat.extend(src)
                    tgt_flat.extend(tgt)
                    src_offsets.append(src_offsets[-1] + len(src))
                    tgt_offsets.append(tgt_offsets[-1] + len(tgt))
                if i % 100000 == 0:
                    print(f"Processed {i} lines")
        print(f"Saving {split} to file")

        pairs = {
            "src": torch.tensor(src_flat, dtype=torch.int16),
            "tgt": torch.tensor(tgt_flat, dtype=torch.int16),
            "src_offsets": torch.tensor(src_offsets, dtype=torch.int64),
            "tgt_offsets": torch.tensor(tgt_offsets, dtype=torch.int64)
        }
        torch.save(pairs, os.path.join(JESC_TOK_DIR, f"{split}.pt"))

    return block_size

def main():
    parser = argparse.ArgumentParser(description="Train tokenizer")
    parser.add_argument(
        "--config", type=str, default="JESC_default.yaml", help="Config file path"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded config file: {args.config}")
    os.makedirs(JESC_TOK_DIR, exist_ok=True)

    block_size = tokenize_dataset(config)
    print(f"block_size (max sequence length across all splits): {block_size}")

    with open(os.path.join(PREPROCESS_CONFIG_PATH, args.config), "r") as f:
        config_raw = yaml.safe_load(f)
    config_raw["block_size"] = block_size
    with open(os.path.join(JESC_TOK_DIR, "config.yaml"), "w") as f:
        yaml.dump(config_raw, f, allow_unicode=True)

if __name__ == "__main__":
    main()