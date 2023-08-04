"""
In the following operations,
    X --- B * d_X * N * N    (node pair embeddings)
    A --- B * N * N          (adjacency matrix)
Output is always B * d_X * N * N.
"""
import torch
from typing import Literal

class PointwiseOp:
    """
    Pointwise operators.
    """
    def __init__(self, op_type: Literal['uv', 'vu', 'uu', 'vv']):
        self.type = op_type

    def __call__(self, X: torch.Tensor):
        if self.type == 'uv':
            return X
        elif self.type == 'vu':
            return X.permute(0, 1, 3, 2)
        elif self.type == 'uu':
            # diag(X) 11^T
            return torch.diagonal(X, dim1=-2, dim2=-1).unsqueeze(-1).expand_as(X)
        elif self.type == 'vv':
            # 11^T diag(X)
            return torch.diagonal(X, dim1=-2, dim2=-1).unsqueeze(-2).expand_as(X)

class GlobalOp:
    """
    Global operators.
    """
    def __init__(self, op_type: Literal['uG', 'vG']):
        self.type = op_type
    
    def __call__(self, X: torch.Tensor):
        if self.type == 'uG':
            # sum over v
            return X.sum(-1, keepdim=True).expand_as(X)
        elif self.type == 'vG':
            # sum over u
            return X.sum(-2, keepdim=True).expand_as(X)

class LocalOp:
    """
    Local operators.
    """
    def __init__(self, op_type: Literal['uL', 'vL']):
        self.type = op_type
    
    def __call__(self, X: torch.Tensor, A: torch.Tensor):
        if self.type == 'uL':
            # XA^T
            return torch.matmul(X, A.permute(0, 2, 1).unsqueeze(1))
        elif self.type == 'vL':
            # AX
            return torch.matmul(A.unsqueeze(1), X)

class BinaryOp:
    """
    Binary operators.
    """
    def __init__(self, op_type: Literal['Nu', 'Nv', 'full']):
        self.type = op_type
    
    def __call__(self, X: torch.Tensor, A: torch.Tensor):
        if self.type == 'Nu':
            # (A \odot X) X
            return torch.matmul(A.unsqueeze(1) * X, X)
        elif self.type == 'Nv':
            # X (A^T \odot X)
            return torch.matmul(X, A.permute(0, 2, 1).unsqueeze(1) * X)
        elif self.type == 'full':
            # XX
            return torch.matmul(X, X)
