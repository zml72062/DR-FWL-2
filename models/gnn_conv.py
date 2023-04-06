"""
GNN layers
"""

import torch
import torch.nn as nn
from torch_scatter import scatter
from pygmmpp.nn.model import MLP
from typing import Optional, Tuple
from .utils import clones

class DR2FWL2Conv(nn.Module):
    """
    An implementation for 2-Distance Restricted FWL(2) GNN layer.

    For every pair (u, v) with d(u, v) <= 2, the layer updates its
    feature W(u, v) by

    W(u, v)  <-  (1 + eps) * W(u, v) +
              M_{11} (u, v) + M_{12} (u, v) +
              M_{21} (u, v) + M_{22} (u, v)

    where

    M_{ij}(u, v) = POOL_{ij} (
        {{ MLP(W(u, w)) * MLP(W(w, v)) : d(u, w) = i and d(w, v) = j }}
    )
    """

    def __init__(self, in_channels: int,
                 hidden_channels: int,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 eps: float = 0.0,
                 eps2: float = 0.0,
                 train_eps: bool = False,
                 norm: Optional[str] = None):
        """
        Args:

        in_channels (int): Input edge-feature and 2-hop-edge-feature channels,
        must be the same for 1-hop and 2-hop edges

        hidden_channels (int): Hidden channels in MLP

        num_layers (int): Number of non-linear layers in MLP, default 1

        dropout (float): Dropout rate after applying ReLU in MLP, default 0.0

        eps, eps2 (float): `eps` value for 1-hop / 2-hop edges (see formula),
        default 0.0 both

        train_eps (bool): Whether to treat `eps` and `eps2` as trainable
        parameters, default False

        norm (Optional[str]): Normalization method for MLP, default None,
        can be None, 'batch_norm' or 'layer_norm'
        """
        super().__init__()

        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps], dtype=torch.float))
            self.eps2 = nn.Parameter(torch.tensor([eps2], dtype=torch.float))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
            self.register_buffer('eps2', torch.Tensor([eps2]))

        proj = nn.Linear(in_channels, in_channels)
        self.projs = clones(proj, 6)

        mlp = MLP(in_channels,
                  hidden_channels,
                  num_layers,
                  in_channels,
                  dropout,
                  norm=norm)
        self.mlps = clones(mlp, 2)

        self.reset_parameters()

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()
        for p in self.projs:
            p.reset_parameters()

    def forward(self,
                edge_attr: torch.Tensor,
                edge_attr2: torch.Tensor,
                triangle_1_1_1: torch.LongTensor,
                triangle_1_1_2: torch.LongTensor,
                triangle_1_2_2: torch.LongTensor,
                triangle_2_2_2: torch.LongTensor,
                inverse_edge_1: torch.LongTensor,
                inverse_edge_2: torch.LongTensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        num_edges: int = edge_attr.shape[0]
        num_edges2: int = edge_attr2.shape[0]

        # For undirected graphs, the direction of edges is unimportant
        ij111, ik111, kj111 = triangle_1_1_1
        ij112, ik112, kj112 = triangle_1_1_2
        ij122, ik122, kj122 = triangle_1_2_2
        ij222, ik222, kj222 = triangle_2_2_2

        multiset_111 = scatter(
            self.projs[0](edge_attr[ik111] + edge_attr[kj111]),
            ij111, dim=0, dim_size=num_edges, reduce='sum'
        )
        multiset_112 = scatter(
            self.projs[1](edge_attr[ik112] + edge_attr2[kj112]),
            ij112, dim=0, dim_size=num_edges, reduce='sum'
        )
        multiset_122 = scatter(
            self.projs[2](edge_attr2[ik122] + edge_attr2[kj122]),
            ij122, dim=0, dim_size=num_edges, reduce='sum'
        )

        edge_attr = self.mlps[0](edge_attr * (1 + self.eps) + multiset_111 + \
                    multiset_112 + multiset_112[inverse_edge_1] + multiset_122)

        multiset_211 = scatter(
            self.projs[3](edge_attr[ij112] + edge_attr[ik112]),
            kj112, dim=0, dim_size=num_edges2, reduce='sum'
        )
        multiset_212 = scatter(
            self.projs[4](edge_attr[ij122] + edge_attr2[kj122]),
            ik122, dim=0, dim_size=num_edges2, reduce='sum'
        )
        multiset_222 = scatter(
            self.projs[5](edge_attr2[ik222] + edge_attr2[kj222]),
            ij222, dim=0, dim_size=num_edges2, reduce='sum'
        )

        edge_attr2 = self.mlps[1](edge_attr2 * (1 + self.eps2) + multiset_211 + \
                     multiset_212 + multiset_212[inverse_edge_2] + multiset_222)
        return edge_attr, edge_attr2






'''
class DR2FWL2Conv(nn.Module):
    """
    An implementation for 2-Distance Restricted FWL(2) GNN layer.

    For every pair (u, v) with d(u, v) <= 2, the layer updates its
    feature W(u, v) by

    W(u, v)  <-  (1 + eps) * W(u, v) +
              M_{11} (u, v) + M_{12} (u, v) +
              M_{21} (u, v) + M_{22} (u, v)

    where

    M_{ij}(u, v) = POOL_{ij} (
        {{ MLP(W(u, w)) * MLP(W(w, v)) : d(u, w) = i and d(w, v) = j }}
    )
    """

    def __init__(self, in_channels: int,
                 hidden_channels: int,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 eps: float = 0.0,
                 eps2: float = 0.0,
                 train_eps: bool = False,
                 norm: Optional[str] = None):
        """
        Args:

        in_channels (int): Input edge-feature and 2-hop-edge-feature channels,
        must be the same for 1-hop and 2-hop edges

        hidden_channels (int): Hidden channels in MLP

        num_layers (int): Number of non-linear layers in MLP, default 1

        dropout (float): Dropout rate after applying ReLU in MLP, default 0.0

        eps, eps2 (float): `eps` value for 1-hop / 2-hop edges (see formula),
        default 0.0 both

        train_eps (bool): Whether to treat `eps` and `eps2` as trainable
        parameters, default False

        norm (Optional[str]): Normalization method for MLP, default None,
        can be None, 'batch_norm' or 'layer_norm'
        """
        super().__init__()
        self.mlps = nn.ModuleList()
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps], dtype=torch.float))
            self.eps2 = nn.Parameter(torch.tensor([eps2], dtype=torch.float))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
            self.register_buffer('eps2', torch.Tensor([eps2]))

        # the 8 mlps are for
        #       in 11->1, transform 1
        #       in 12->1, transform 1
        #       in 12->1, transform 2
        #       in 22->1, transform 2
        #       in 11->2, transform 1
        #       in 12->2, transform 1
        #       in 12->2, transform 2
        #       in 22->2, transform 2
        for _ in range(6):
            self.mlps.append(MLP(in_channels,
                                 hidden_channels,
                                 num_layers,
                                 in_channels,
                                 dropout,
                                 norm=norm))

        self.reset_parameters()

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self,
                edge_attr: torch.Tensor,
                edge_attr2: torch.Tensor,
                triangle_1_1_1: torch.LongTensor,
                triangle_1_1_2: torch.LongTensor,
                triangle_1_2_2: torch.LongTensor,
                triangle_2_2_2: torch.LongTensor,
                inverse_edge_1: torch.LongTensor,
                inverse_edge_2: torch.LongTensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        num_edges: int = edge_attr.shape[0]
        num_edges2: int = edge_attr2.shape[0]

        # For undirected graphs, the direction of edges is unimportant
        ij111, ik111, kj111 = triangle_1_1_1
        ij112, ik112, kj112 = triangle_1_1_2
        ij122, ik122, kj122 = triangle_1_2_2
        ij222, ik222, kj222 = triangle_2_2_2

        multiset_111 = scatter(
            self.mlps[0](edge_attr[ik111] + edge_attr[kj111]),
            ij111, dim=0, dim_size=num_edges, reduce='sum'
        )
        multiset_112 = scatter(
            self.mlps[1](edge_attr[ik112] + edge_attr2[kj112]),
            ij112, dim=0, dim_size=num_edges, reduce='sum'
        )
        multiset_122 = scatter(
            self.mlps[2](edge_attr2[ik122] + edge_attr2[kj122]),
            ij122, dim=0, dim_size=num_edges, reduce='sum'
        )

        edge_attr = edge_attr * (1 + self.eps) + multiset_111 + \
                    multiset_112 + multiset_112[inverse_edge_1] + multiset_122

        multiset_211 = scatter(
            self.mlps[3](edge_attr[ij112] + edge_attr[ik112]),
            kj112, dim=0, dim_size=num_edges2, reduce='sum'
        )
        multiset_212 = scatter(
            self.mlps[4](edge_attr[ij122] + edge_attr2[kj122]),
            ik122, dim=0, dim_size=num_edges2, reduce='sum'
        )
        multiset_222 = scatter(
            self.mlps[5](edge_attr2[ik222] + edge_attr2[kj222]),
            ij222, dim=0, dim_size=num_edges2, reduce='sum'
        )

        edge_attr2 = edge_attr2 * (1 + self.eps2) + multiset_211 + \
                     multiset_212 + multiset_212[inverse_edge_2] + multiset_222
        return edge_attr, edge_attr2


'''
