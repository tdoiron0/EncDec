import os

from constants import WMT14_RAW_DIR, WMT14_CORPUS_NAME, CORPORA_PATH

outfile = os.path.join(CORPORA_PATH, WMT14_CORPUS_NAME)
os.makedirs(CORPORA_PATH, exist_ok=True)

splits = ["train"]
for split in splits:
    with open(os.path.join(WMT14_RAW_DIR, split), 'r', encoding="utf-8") as fin, open(outfile, 'w', encoding="utf-8") as fout:
        for line in fin:
            cols = line.rstrip("\n").split("\t")
            for col in cols:
                fout.write(f"{col}\n")