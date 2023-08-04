from torch import Tensor
import torch.nn as nn
from . import utils
from typing import Callable, Literal, Optional

class LocalFWL2(nn.Module):
    def __init__(self, channels: int,
                 num_layers: int,
                 layer: Callable,
                 node_in_channels: int,
                 edge_in_channels: Optional[int] = None,
                 norm: Literal['no', 'instance', 'batch'] = 'no'):
        super().__init__()
        self.init = utils.InitialEmbedding(channels, node_in_channels, 
                                           edge_in_channels)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(layer(channels, norm))
        
    def forward(self, A: Tensor, X: Tensor, E: Optional[Tensor] = None) -> Tensor:
        X = self.init(X, E)
        for layer in self.layers:
            X = layer(X, A)
        return X

class Pooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
    
    def forward(self, X: Tensor) -> Tensor:
        return self.lin(X.sum(-1).sum(-1))
