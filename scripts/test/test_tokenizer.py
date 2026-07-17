import os

from constants import TOKENIZER_CONFIG_PATH, TOKENIZERS_PATH
from src.tokenizer.tokenizer import Tokenizer

def piece_token_pairs(sentence: str, tok: Tokenizer):
    pieces = tok.encode(sentence, out_type=str)
    tokens = tok.encode(sentence, out_type=int)
    return list(zip(pieces, tokens))


# tok = Tokenizer.train(os.path.join(TOKENIZER_CONFIG_PATH, "de-en.yaml"))
tok = Tokenizer.load(os.path.join(TOKENIZERS_PATH, "wmt14-1.model"))

src = "Eine Blackbox im Auto?"
tgt = "A black box in your car?"

print(src)
print(piece_token_pairs(src, tok))
print(tgt)
print(piece_token_pairs(tgt, tok))