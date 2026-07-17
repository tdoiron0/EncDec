import os
import shutil
from datasets import load_dataset

from constants import WMT14_RAW_DIR

ds = load_dataset("wmt/wmt14", "de-en")

splits = ['train', 'validation', 'test']

if os.path.exists(WMT14_RAW_DIR):
    shutil.rmtree(WMT14_RAW_DIR)
os.makedirs(WMT14_RAW_DIR, exist_ok=True)

for split in splits:
    print(f"Downloading {split}")
    i = 0
    with open(os.path.join(WMT14_RAW_DIR, f"{split}"), "w", encoding="utf-8") as f:
        for it in ds[split]:
            f.write(f"{it['translation']['en']}\t{it['translation']['de']}\n")

            if i % 100000 == 0:
                print(f"Downloaded {i} lines")
            i += 1
    print(f"Downloaded {i} lines")