import os
import torch

from src.datasets.JESC_dataset import JESCDataset
from src.datasets.wmt14_dataset import WMT14Dataset
from constants.constants import JESC_NAME, WMT14_NAME

DATASET_INDEX: dict[str, torch.utils.data.Dataset] = {
    JESC_NAME: JESCDataset,
    WMT14_NAME: WMT14Dataset
}