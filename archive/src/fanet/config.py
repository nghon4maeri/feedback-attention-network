"""YAML configuration loader for FANet experiments."""
import os

import yaml


DEFAULTS = {
    "dataset": {
        "root": "data/sessile-main-Kvasir-SEG",
        "image_size": [256, 256],
    },
    "train": {
        "batch_size": 2,
        "epochs": 500,
        "lr": 1e-4,
        "seed": 42,
        "augmentation": {
            "rotate_limit": 35,
            "rotate_p": 0.3,
            "hflip_p": 0.3,
            "vflip_p": 0.3,
            "dropout_p": 0.3,
        },
    },
    "eval": {
        "num_iter": 10,
    },
    "paths": {
        "checkpoint": "checkpoints/checkpoint.pth",
        "train_log": "logs/train_log.txt",
        "results": "results",
    },
}


def load_config(path):
    cfg = {}
    if path is not None and os.path.exists(path):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    merged = {}
    for key in DEFAULTS:
        merged[key] = {**DEFAULTS[key], **(cfg.get(key) or {})}
    return merged
