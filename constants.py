import sys, os
import torch
from typing import Type, Callable

'''====================GENERAL PROJECT STRUCTURE===================='''
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR))
RUNS_PATH = os.path.join(PROJECT_ROOT, "runs")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config")

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "src", "datasets", "data-raw")
PROC_DATA_DIR = os.path.join(PROJECT_ROOT, "src", "datasets", "data-processed")

'''====================CONFIGS===================='''
TOKENIZER_CONFIG_PATH = os.path.join(CONFIG_PATH, "tokenizer")
TRAIN_CONFIGS_PATH = os.path.join(CONFIG_PATH, "train")
PREPROCESS_CONFIG_PATH = os.path.join(CONFIG_PATH, "preprocess")

'''====================TOKENIZERS===================='''
TOKENIZERS_PATH = os.path.join(PROJECT_ROOT, "tokenizers")
CORPORA_PATH = os.path.join(TOKENIZERS_PATH, "corpora")

BOS_TOKEN = 1
EOS_TOKEN = 2
PAD_TOKEN = 3

'''====================MODELS===================='''
ENC_DEC_NAME = "enc-dec"

'''====================TRAINERS===================='''
DEFAULT_TRAINER_NAME = "default"

'''====================DATASETS===================='''
JESC_CORPUS_NAME = "jesc-train"
JESC_CORPUS_PATH = os.path.join(CORPORA_PATH, JESC_CORPUS_NAME)
JESC_RAW_NAME = "jesc-split"
JESC_RAW_DIR = os.path.join(RAW_DATA_DIR, "JESC-split")
JESC_TOK_NAME = "jesc-split-tokens"
JESC_TOK_DIR = os.path.join(PROC_DATA_DIR , "JESC-split-tokens")

WMT14_CORPUS_NAME = "wmt14-train"
WMT14_CORPUS_PATH = os.path.join(CORPORA_PATH, WMT14_CORPUS_NAME)
WMT14_RAW_NAME = "wmt14-de-en"
WMT14_RAW_DIR = os.path.join(RAW_DATA_DIR, "wmt14-de-en")
WMT14_TOK_NAME = "wmt14-de-en-tokens"
WMT14_TOK_DIR = os.path.join(PROC_DATA_DIR, "wmt14-de-en-tokens")

DATASET_DIRS: dict[str, str] = {
    JESC_RAW_NAME: JESC_RAW_DIR, 
    JESC_TOK_NAME: JESC_TOK_DIR,
    WMT14_RAW_NAME: WMT14_RAW_DIR,
    WMT14_TOK_NAME: WMT14_TOK_DIR
}

'''====================SCHEDULERS===================='''
SEQUENTIAL_LR_SCHEDULER_NAME = "sequential-lr"

'''====================OPTIMIZERS===================='''
ADAMW_NAME = "adamw"

'''====================DATATYPES===================='''
FLOAT16 = "fp16"
BFLOAT16 = "bf16"

DTYPE_INDEX: dict[str, torch.dtype] = {
    FLOAT16: torch.float16,
    BFLOAT16: torch.bfloat16,
}