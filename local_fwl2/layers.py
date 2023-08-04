"""
Implementations for SSWL, SSWL+, LFWL(2) and SLFWL(2) layers.
"""
import torch
import torch.nn as nn
from . import operators as op
from . import utils
from typing import Literal, Optional

class OpLayer(nn.Module):
    def __init__(self, in_channels: int, 
                 out_channels: int,
                 op_type: Literal['uv', 'vu', 'uu', 'vv', 'uG', 'vG',
                                  'uL', 'vL', 'Nu', 'Nv', 'full'],
                 norm: Literal['no', 'instance', 'batch'] = 'no',
                 eps: float = 0.0):
        super().__init__()
        self.nn = utils.ReLULin(in_channels, out_channels, norm)
        self.op_type = op_type
        self.eps = eps
        if op_type in {'uv', 'vu', 'uu', 'vv'}:
            self.op = op.PointwiseOp(op_type)
        elif op_type in {'uG', 'vG'}:
            self.op = op.GlobalOp(op_type)
        elif op_type in {'uL', 'vL'}:
            self.op = op.LocalOp(op_type)
        elif op_type in {'Nu', 'Nv', 'full'}:
            self.op = op.BinaryOp(op_type)

    def forward(self, X: torch.Tensor, 
                A: Optional[torch.Tensor] = None) -> torch.Tensor:
        after_nn = self.nn(X)
        if self.op_type in {'uv', 'vu', 'uu', 'vv', 'uG', 'vG'}:
            after_op = self.op(after_nn)
        elif self.op_type in {'uL', 'vL', 'Nu', 'Nv', 'full'}:
            after_op = self.op(after_nn, A)
        return (1 + self.eps) * after_nn + after_op

class SSWLLayer(nn.Module):
    def __init__(self, channels: int,
                 norm: Literal['no', 'instance', 'batch'] = 'no'):
        super().__init__()
        self.uL = OpLayer(channels, channels, 'uL', norm)
        self.vL = OpLayer(channels, channels, 'vL', norm)
        self.mlp = nn.Sequential(utils.ReLULin(channels, channels, norm),
                                 nn.Conv2d(channels, channels, (1, 1)))
        
    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        agg = self.uL(X, A) + self.vL(X, A)
        return agg + self.mlp(agg)
    
class SSWLPlusLayer(nn.Module):
    def __init__(self, channels: int,
                 norm: Literal['no', 'instance', 'batch'] = 'no'):
        super().__init__()
        self.uL = OpLayer(channels, channels, 'uL', norm)
        self.vL = OpLayer(channels, channels, 'vL', norm)
        self.vv = OpLayer(channels, channels, 'vv', norm)
        self.mlp = nn.Sequential(utils.ReLULin(channels, channels, norm),
                                 nn.Conv2d(channels, channels, (1, 1)))
        
    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        agg = self.uL(X, A) + self.vL(X, A) + self.vv(X)
        return agg + self.mlp(agg)
    
class LFWLLayer(nn.Module):
    def __init__(self, channels: int,
                 norm: Literal['no', 'instance', 'batch'] = 'no'):
        super().__init__()
        self.Nv = OpLayer(channels, channels, 'Nv', norm)
        self.mlp = nn.Sequential(utils.ReLULin(channels, channels, norm),
                                 nn.Conv2d(channels, channels, (1, 1)))
        
    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        agg = self.Nv(X, A)
        return agg + self.mlp(agg)
  
class SLFWLLayer(nn.Module):
    def __init__(self, channels: int,
                 norm: Literal['no', 'instance', 'batch'] = 'no'):
        super().__init__()
        self.Nv = OpLayer(channels, channels, 'Nv', norm)
        self.Nu = OpLayer(channels, channels, 'Nu', norm)
        self.mlp = nn.Sequential(utils.ReLULin(channels, channels, norm),
                                 nn.Conv2d(channels, channels, (1, 1)))
        
    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        agg = self.Nu(X, A) + self.Nv(X, A)
        return agg + self.mlp(agg)