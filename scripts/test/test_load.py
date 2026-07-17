import os
import torch
from pprint import pprint

from src.model.enc_dec import EncoderDecoder
from src.tokenizer.tokenizer import Tokenizer
from constants import TOKENIZERS_PATH, RUNS_PATH
from utils import load_config

run_path = os.path.join(RUNS_PATH, "de-en-medium-2")

checkpoint_path = os.path.join(run_path, "checkpoints", "checkpoint_epoch_1_iter_17524.pt")
checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

config = load_config(os.path.join(run_path, "train_config.yaml"))
tokenizer = Tokenizer.load(os.path.join(run_path, "tokenizer.model"))
config.vocab_size = tokenizer.vocab_size()
pprint(config)

model = EncoderDecoder(config)

# Gutach: Increased safety for pedestrians
src = "Eine Blackbox im Auto?"
print(list(zip(tokenizer.encode(src, out_type=str), tokenizer.encode(src, out_type=int))))
src_ids = [tokenizer.bos_id()] + tokenizer.encode(src) + [tokenizer.eos_id()]
src = torch.tensor([src_ids], dtype=torch.long)

model.load(checkpoint)

out = model.generate(
    src, 
    200, 
    bos_id=tokenizer.bos_id(), 
    eos_id=tokenizer.eos_id()
)

print(tokenizer.decode(out))