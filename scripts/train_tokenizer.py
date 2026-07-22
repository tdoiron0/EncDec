import os

from constants import TOKENIZER_CONFIG_PATH, TOKENIZERS_PATH
from src.tokenizer.tokenizer import Tokenizer

tok = Tokenizer.train(os.path.join(TOKENIZER_CONFIG_PATH, "de-en.yaml"))