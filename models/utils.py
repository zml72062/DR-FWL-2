import torch.nn as nn
from copy import deepcopy as c
from typing import Optional


def clones(module: nn.Module,
           N: int) -> Optional[nn.ModuleList]:
    r"""Layer clone function, used for concise code writing. If input is None, simply return None.
    Args:
        module (nn.Module): Module want to clone.
        N (int): Clone times.
    """
    if module is None:
        return module
    else:
        return nn.ModuleList(c(module) for _ in range(N))
