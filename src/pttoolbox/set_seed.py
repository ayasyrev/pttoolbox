import os
import random

import numpy as np
import torch


def set_seed(
    seed_num: int = 42,
    seed_pythonhash: bool = True,
    seed_random: bool = True,
    seed_numpy: bool = True,
    seed_torch: bool = True,
    torch_benchmark: bool = True,
    torch_deterministic: bool = False,
) -> None:
    """Set seeds.

    Args:
        seed_num: seed number, default 42
        seed_pythonhash: set hash seed, default True
        seed_random: set random seed, default True
        seed_numpy: set numpy seed, default True
        seed_torch: set torch seed, default True
        torch_benchmark: set torch benchmark, default True
        torch_deterministic: set torch deterministic, default False
    """
    if seed_pythonhash:
        os.environ["PYTHONHASHSEED"] = str(seed_num)
    if seed_random:
        random.seed(seed_num)
    if seed_numpy:
        np.random.seed(seed_num)
    if seed_torch:
        torch.manual_seed(seed_num)
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)

    torch.backends.cudnn.benchmark = torch_benchmark
    torch.backends.cudnn.deterministic = torch_deterministic
