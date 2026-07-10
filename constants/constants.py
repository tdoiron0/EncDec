import sys, os
import torch
from typing import Type, Callable

'''====================UTIL===================='''
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config")
TOKENIZER_CONFIG_PATH = os.path.join(CONFIG_PATH, "tokenizer")
TRAINING_CONFIG_PATH = os.path.join(CONFIG_PATH, "training")
PREPROCESS_CONFIG_PATH = os.path.join(CONFIG_PATH, "preprocess")
RUNS_PATH = os.path.join(PROJECT_ROOT, "runs")

'''====================TOKENIZERS===================='''
TOKENIZERS_PATH = os.path.join(PROJECT_ROOT, "tokenizers")
CORPORA_PATH = os.path.join(TOKENIZERS_PATH, "corpora")

BOS_TOKEN = 1
EOS_TOKEN = 2
PAD_TOKEN = 3

'''====================DATASETS===================='''
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "src", "datasets", "data-raw")
PROC_DATA_DIR = os.path.join(PROJECT_ROOT, "src", "datasets", "data-processed")

JESC_NAME = "jesc-split"
JESC_RAW_DIR = os.path.join(RAW_DATA_DIR, "JESC-split")
JESC_PROC_PATH = os.path.join(PROC_DATA_DIR , "JESC-split-tokens")
JESC_CORPUS_NAME = "jesc-train"

WMT14_NAME = "wmt14-de-en"
WMT14_RAW_DIR = os.path.join(RAW_DATA_DIR, "wmt14-de-en")
WMT14_PROC_PATH = os.path.join(PROC_DATA_DIR, "wmt14-de-en-tokens")
WMT14_CORPUS_NAME = "wmt14-train"

DATASET_PATHS: dict[str, str] = {
    JESC_NAME: JESC_PROC_PATH,
    WMT14_NAME: WMT14_PROC_PATH
}