import os
import sentencepiece as spm
from dataclasses import dataclass
import json

from constants import TOKENIZER_CONFIG_PATH, PROJECT_ROOT, TOKENIZERS_PATH, CORPORA_PATH
from utils import load_config

class Tokenizer:
    def __init__(self, sp: spm.SentencePieceProcessor):
        self.sp = sp
    
    @classmethod
    def load(cls, model_filepath):
        sp = spm.SentencePieceProcessor()
        sp.load(model_filepath)
        return cls(sp)

    @classmethod
    def train(cls, config_filepath):
        config = load_config(config_filepath)
        
        corpus_paths = [os.path.join(CORPORA_PATH, it) for it in config.raws]
        out_path = os.path.join(TOKENIZERS_PATH, config.out_name)

        spm.SentencePieceTrainer.train(
            input=",".join(corpus_paths),
            model_prefix=out_path,
            vocab_size=config.vocab_size,
            character_coverage=config.character_coverage,
            model_type=config.model_type,
            unk_id=config.unk_id, bos_id=config.bos_id, eos_id=config.eos_id, pad_id=config.pad_id,
            input_sentence_size=config.input_sentence_size,
            shuffle_input_sentence=config.shuffle_input_sentence,
            num_threads=os.cpu_count(),
        )

        return cls.load(f"{out_path}.model")
    
    def encode(self, text: str, out_type=int) -> list[int]:
        return self.sp.encode(text, out_type=out_type)

    def encode_batch(self, texts: list[str], out_type=int) -> list[list[int]]:
        return self.sp.encode(texts, out_type=out_type, num_threads=os.cpu_count() or 1)
    
    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)
    
    def bos_id(self) -> int:
        return self.sp.bos_id()

    def eos_id(self) -> int:
        return self.sp.eos_id()
    
    def pad_id(self) -> int:
        return self.sp.pad_id()
    
    def unk_id(self) -> int:
        return self.sp.unk_id()
    
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()