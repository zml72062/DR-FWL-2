import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj, to_dense_batch
from typing import Optional, Tuple, Literal

def to_dense(x: torch.Tensor, edge_index: torch.LongTensor,
             edge_attr: Optional[torch.Tensor] = None,
             batch: Optional[torch.Tensor] = None) \
    -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Given a PyG batch (x, edge_index, edge_attr, batch), return a dense
    batch with 
            A --- B * N * N         (adjacency matrix);
            X --- B * N * d_X       (node feature);
            E --- B * N * N * d_E   (edge feature, may be None);
    """
    A = to_dense_adj(edge_index, batch)
    X, _ = to_dense_batch(x, batch)
    if edge_attr is None:
        E = None
    else:
        E = to_dense_adj(edge_index, batch, edge_attr)
    return A, X, E

class InitialEmbedding(nn.Module):
    """
    Simple learnable module to generate dense embedding. Output shape is
    B * d * N * N, d is embedding dim.
    """
    def __init__(self, out_channels: int,
                 node_in_channels: int, 
                 edge_in_channels: Optional[int] = None):
        super().__init__()
        self.out_channels = out_channels
        self.node_transform = nn.Conv2d(node_in_channels, out_channels, (1, 1))
        if edge_in_channels is None:
            self.edge_transform = None
        else:
            self.edge_transform = nn.Conv2d(edge_in_channels, out_channels, (1, 1))
    
    def forward(self, X: torch.Tensor, E: Optional[torch.Tensor] = None) \
        -> torch.Tensor:
        """
        return "ReLU(lin1(X_u + X_v) + lin2(E_uv))" or "ReLU(lin1(X_u + X_v))".
        """
        from_node = self.node_transform(
            (X.unsqueeze(1) + X.unsqueeze(2)).permute(0, 3, 1, 2))
        if E is None or self.edge_transform is None:
            return nn.ReLU()(from_node)
        else:
            from_edge = self.edge_transform(E.permute(0, 3, 1, 2))
            return nn.ReLU()(from_node + from_edge)

class ReLULin(nn.Module):
    """
    ReLU(lin(x)) with optional normalization.
    """
    def __init__(self, in_channels: int,
                 out_channels: int,
                 norm: Literal['no', 'instance', 'batch'] = 'no'):
        super().__init__()
        self.lin = nn.Conv2d(in_channels, out_channels, (1, 1))
        if norm == 'no':
            self.norm = nn.Identity()
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.ReLU()(self.norm(self.lin(x)))
