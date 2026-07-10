import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from jaxtyping import Float, Int 

from transformer import GenericTransformer
from encoder import Encoder 
from decoder import Decoder

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.model_config import ModelConfig

_MODEL_REGISTRY: dict[str, type[GenericTransformer]] = {
    "generic": GenericTransformer,
    "encoder": Encoder,
    "decoder": Decoder,
}

def get_model(config: ModelConfig) -> nn.Module:
    if config.model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{config.model_type}'. "
            f"Choose from: {list(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[config.model_type](config)