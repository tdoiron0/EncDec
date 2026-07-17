import os
import sys
import json
import random
from dataclasses import asdict

import numpy as np
import torch
from typing import List
import logging
from types import SimpleNamespace
import yaml


def get_best_device() -> str:
    """
    Get the best available device in order of preference: cuda, mps, cpu.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def setup_logging():
    """monotonous bookkeeping"""
    work_dir = "logs"
    # create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)
    # log the args (if any)
    with open(os.path.join(work_dir, "args.txt"), "w") as f:
        f.write(" ".join(sys.argv))


def format_review(row):
    return {
        "text": f"{row['translation']['eng']}[SEP]{row['translation']['engyay']}[END]"
    }

def load_config(filepath):
    def to_namespace(data):
        if isinstance(data, dict):
            return SimpleNamespace(
                **{key: to_namespace(value) for key, value in data.items()}
            )
        elif isinstance(data, list):
            return [to_namespace(item) for item in data]
        return data
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
    return to_namespace(data)

from argparse import Namespace

def namespace_to_dict(obj):
    if hasattr(obj, "__dict__"):
        return {
            k: namespace_to_dict(v)
            for k, v in vars(obj).items()
        }
    elif isinstance(obj, dict):
        return {
            k: namespace_to_dict(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [namespace_to_dict(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(namespace_to_dict(v) for v in obj)
    elif isinstance(obj, set):
        return {namespace_to_dict(v) for v in obj}
    else:
        return obj
    
def resolve(x: str, attempts: list[str], isfile: bool = True) -> str:
    if isfile:
        filepath = None
        for it in attempts:
            if os.path.isfile(it):
                return filepath
        return None
    else:
        dir = None
        for it in attempts:
            if os.path.isdir(it):
                return dir
        return None