import os

from constants.constants import WMT14_RAW_DIR

split = 'test'

with open(os.path.join(WMT14_RAW_DIR, f"{split}"), "r", encoding="utf-8") as f:
    for line in f:
        en, de = line.rstrip("\n").split("\t")
        print(f"English: {en}")
        print(f"German: {de}")