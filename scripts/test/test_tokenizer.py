import os

from constants.constants import TOKENIZER_CONFIG_PATH, TOKENIZERS_PATH
from src.tokenizer.tokenizer import Tokenizer

def piece_token_pairs(sentence: str, tok: Tokenizer):
    pieces = tok.encode(sentence, out_type=str)
    tokens = tok.encode(sentence, out_type=int)
    return list(zip(pieces, tokens))


# tok = Tokenizer.train(os.path.join(TOKENIZER_CONFIG_PATH, "de-en.yaml"))
tok = Tokenizer.load(os.path.join(TOKENIZERS_PATH, "wmt14-1.model"))

print("Hello my name is Trent!")
print(piece_token_pairs("Hello my name is Trent!", tok))
print("Gutach: Noch mehr Sicherheit für Fußgänger")
print(piece_token_pairs("Gutach: Noch mehr Sicherheit für Fußgänger", tok))