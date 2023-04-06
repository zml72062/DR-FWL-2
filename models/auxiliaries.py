"""
Auxiliaries model components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TwoComponentReLU(torch.nn.Module):
    """
    Apply ReLU component-wise for Tuple[torch.Tensor, torch.Tensor]
    """

    def __init__(self):
        super().__init__()

    def forward(self,
                edge_attr: torch.Tensor,
                edge_attr2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return F.relu(edge_attr), F.relu(edge_attr2)


class TwoComponentLinear(torch.nn.Module):
    """
    Apply nn.Linear component-wise for Tuple[torch.Tensor, torch.Tensor]
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = nn.Linear(in_channels, out_channels)
        self.lin2 = nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, edge_attr: torch.Tensor,
                edge_attr2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lin1(edge_attr), self.lin2(edge_attr2)




class TwoComponentEmbedding(torch.nn.Module):
    """
    Apply nn.Embeeding component-wise for Tuple[torch.Tensor, torch.Tensor]
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = nn.Embedding(in_channels, out_channels)
        self.lin2 = nn.Embedding(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, edge_attr: torch.Tensor,
                edge_attr2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lin1(edge_attr), self.lin2(edge_attr2)