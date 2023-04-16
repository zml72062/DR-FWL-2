"""
Auxiliaries model components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultipleComponentLinear(nn.Module):
    r"""Linear projection with ELU activation for multiple input components.
        Notes all components share the same set of parameters.
    Args:
        in_channels (int): Input size.
        out_channels (int): Output size.
        drop_prob (float): Dropout probability.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 drop_prob: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.drop_prob = drop_prob
        self.linear = nn.Linear(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self,
                x_tuple: Tuple) -> Tuple:
        out_list = [F.elu(self.linear(x)) for x in x_tuple]
        return tuple(out_list)

