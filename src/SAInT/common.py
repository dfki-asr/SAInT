import os
import json
import random
import numpy as np
from fastai.tabular.all import torch


def set_seed(random_seed: int = 123456):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def makedirs(dirname: str):
    os.makedirs(dirname, exist_ok=True)


def exists(path: str):
    return os.path.exists(path)


def load_json_dict(filepath: str) -> dict:
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            loaded_dict = json.load(file)
            return loaded_dict
    return {}


def rel_to_abs_path(rel_path):
    return os.path.abspath(rel_path)
