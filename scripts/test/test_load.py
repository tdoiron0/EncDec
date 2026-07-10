import os
import torch
from pprint import pprint

from src.model.enc_dec import EncoderDecoder
from src.tokenizer.tokenizer import Tokenizer
from constants.constants import TOKENIZERS_PATH


checkpoint_path = "/home/trent/CodeProjects/CS-4644/Models/Jap2Eng/runs/de-en-medium-2/checkpoints/checkpoint_epoch_4_iter_17524.pt"
checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

model = EncoderDecoder(checkpoint["config"])
tokenizer = Tokenizer.load(os.path.join(TOKENIZERS_PATH, "wmt14-1.model"))

pprint(checkpoint["config"])

# Gutach: Increased safety for pedestrians
src = "Bekannt ist der Büchner-Preisträger vor allem als Prosaautor, Theatertexte sind in seinem Werk rar."
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