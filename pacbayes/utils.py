import json
import numpy as np
import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def experiment_name(hidden, random_labels):
    """Table 1 naming: T-600, T-300-300, ect."""
    return ("R-" if random_labels else "T-") + "-".join(str(h) for h in hidden)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
