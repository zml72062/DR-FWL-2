import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter
from pygmmpp.nn.model import MLP
from typing import Optional, Tuple

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
            self.eps = torch.tensor([eps], dtype=torch.float)
            self.eps2 = torch.tensor([eps2], dtype=torch.float)
        # the 8 mlps are for
        #       in 11->1, transform 1
        #       in 12->1, transform 1
        #       in 12->1, transform 2
        #       in 22->1, transform 2
        #       in 11->2, transform 1
        #       in 12->2, transform 1
        #       in 12->2, transform 2
        #       in 22->2, transform 2
        for _ in range(8):
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
            self.mlps[0](edge_attr[ik111]) * self.mlps[0](edge_attr[kj111]),
            ij111, dim=0, dim_size=num_edges, reduce='sum'
        )
        multiset_112 = scatter(
            self.mlps[1](edge_attr[ik112]) * self.mlps[2](edge_attr2[kj112]),
            ij112, dim=0, dim_size=num_edges, reduce='sum'
        )
        multiset_122 = scatter(
            self.mlps[3](edge_attr2[ik122]) * self.mlps[3](edge_attr2[kj122]),
            ij122, dim=0, dim_size=num_edges, reduce='sum'
        )
        
        edge_attr = edge_attr * (1 + self.eps) + multiset_111 + \
        multiset_112 + multiset_112[inverse_edge_1] + multiset_122

        multiset_211 = scatter(
            self.mlps[4](edge_attr[ij112]) * self.mlps[4](edge_attr[ik112]),
            kj112, dim=0, dim_size=num_edges2, reduce='sum'
        )
        multiset_212 = scatter(
            self.mlps[5](edge_attr[ij122]) * self.mlps[6](edge_attr2[kj122]),
            ik122, dim=0, dim_size=num_edges2, reduce='sum'
        )
        multiset_222 = scatter(
            self.mlps[7](edge_attr2[ik222]) * self.mlps[7](edge_attr2[kj222]),
            ij222, dim=0, dim_size=num_edges2, reduce='sum'
        )
        
        edge_attr2 = edge_attr2 * (1 + self.eps2) + multiset_211 + \
        multiset_212 + multiset_212[inverse_edge_2] + multiset_222

        return edge_attr, edge_attr2
    
class TwoComponentReLU(torch.nn.Module):
    """
    Apply ReLU component-wise for Tuple[torch.Tensor, torch.Tensor]
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, edge_attr: torch.Tensor, 
                edge_attr2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return F.relu(edge_attr), F.relu(edge_attr2)

class TwoComponentLinear(torch.nn.Module):
    """
    Apply nn.Linear component-wise for Tuple[torch.Tensor, torch.Tensor]
    """
    def __init__(self, in_channels: int,
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
        


class DR2FWL2Kernel(torch.nn.Module):
    """
    Define a 2-Distance Restricted FWL(2) GNN kernel by stacking `DR2FWL2Conv`
    layers. This kernel can be further combined with node-level/graph-level
    pooling layers as it is applied on different tasks.
    """
    def __init__(self, 
                 in_channels: int, 
                 mlp_hidden_channels: int,
                 num_layers: int, 
                 dropout: float = 0, 
                 residual: Optional[str] = None, 
                 norm: Optional[str] = None, 
                 relu_first: bool = False, 
                 mlp_num_layers: int = 1,
                 mlp_dropout: float = 0.0,
                 eps: float = 0.0,
                 eps2: float = 0.0,
                 train_eps: bool = False,
                 mlp_norm: Optional[str] = None):
        """
        Args:

        in_channels (int): Input edge-feature and 2-hop-edge-feature channels,
        must be the same for 1-hop and 2-hop edges

        mlp_hidden_channels (int): Hidden channels in MLP

        num_layers (int): Number of `DR2FWL2Conv` layers

        dropout (float): Dropout rate after applying a `DR2FWL2Conv` layer and
        ReLU, default 0.0

        residual (Optional[str]): Whether, and how to use skip connection among
        `DR2FWL2Conv` layers, default None, can be None, 'cat' or 'add'

        norm (Optional[str]): Normalization method after applying a `DR2FWL2Conv`
        layer, default None, can be None, 'batch_norm' or 'layer_norm'

        relu_first (bool): Whether to apply ReLU before normalization, default
        False.

        mlp_num_layers (int): Number of non-linear layers in MLP, default 1

        mlp_dropout (float): Dropout rate after applying ReLU in MLP, default 0.0

        eps, eps2 (float): `eps` value for 1-hop / 2-hop edges (see formula),
        default 0.0 both

        train_eps (bool): Whether to treat `eps` and `eps2` as trainable
        parameters, default False

        mlp_norm (Optional[str]): Normalization method for MLP, default None,
        can be None, 'batch_norm' or 'layer_norm'
        
        """
        super().__init__()

        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(DR2FWL2Conv(
            in_channels, mlp_hidden_channels, mlp_num_layers,
            mlp_dropout, eps, eps2, train_eps, mlp_norm))

        if residual != 'cat':
            self.lins.append(torch.nn.Linear(
                in_channels, in_channels)) # for 1-hop
            self.lins.append(torch.nn.Linear(
                in_channels, in_channels)) # for 2-hop
        else:
            self.lins.append(torch.nn.Linear(
                in_channels*num_layers, in_channels)) # for 1-hop
            self.lins.append(torch.nn.Linear(
                in_channels*num_layers, in_channels)) # for 2-hop

        self.norms1 = torch.nn.ModuleList()
        for _ in range(num_layers):
            if norm is None:
                self.norms1.append(torch.nn.Identity())
            elif norm == 'batch_norm':
                self.norms1.append(torch.nn.BatchNorm1d(in_channels))
            elif norm == 'layer_norm':
                self.norms1.append(torch.nn.LayerNorm(in_channels))
        
        self.norms2 = torch.nn.ModuleList()
        for _ in range(num_layers):
            if norm is None:
                self.norms2.append(torch.nn.Identity())
            elif norm == 'batch_norm':
                self.norms2.append(torch.nn.BatchNorm1d(in_channels))
            elif norm == 'layer_norm':
                self.norms2.append(torch.nn.LayerNorm(in_channels))

        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.dropout = dropout
        self.relu_first = relu_first
        self.residual = residual

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms1:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()
        for norm in self.norms2:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

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
        if self.residual is not None:
            emb_list = []
            emb_list2 = []
        
        for i in range(self.num_layers):
            edge_attr, edge_attr2 = self.lins[i](
                edge_attr,
                edge_attr2,
                triangle_1_1_1,
                triangle_1_1_2,
                triangle_1_2_2,
                triangle_2_2_2,
                inverse_edge_1,
                inverse_edge_2
            )
            if self.relu_first:
                edge_attr, edge_attr2 = TwoComponentReLU()(edge_attr, edge_attr2)
            edge_attr, edge_attr2 = (self.norms1[i](edge_attr),
                                     self.norms2[i](edge_attr2))
            if not self.relu_first:
                edge_attr, edge_attr2 = TwoComponentReLU()(edge_attr, edge_attr2)
            edge_attr, edge_attr2 = F.dropout(
                edge_attr, p=self.dropout, training=self.training), F.dropout(
                edge_attr2, p=self.dropout, training=self.training
                )
            
            if self.residual is not None:
                emb_list.append(edge_attr)
                emb_list2.append(edge_attr2)
        
        if self.residual is None:
            return self.lins[-2](edge_attr), self.lins[-1](edge_attr2)
        elif self.residual == 'add':
            return self.lins[-2](sum(emb_list)), self.lins[-1](sum(emb_list2))
        elif self.residual == 'cat':
            return (self.lins[-2](torch.cat(emb_list, dim=-1)),
                    self.lins[-1](torch.cat(emb_list2, dim=-1)))
        
class NodeLevelPooling(nn.Module):
    """
    Define a node-level pooling function for 2-DRFWL(2):

    W(u) = \sum_{d(u, w) <= 2} W(u, w)
    """
    def __init__(self):
        super().__init__()

    def forward(self, edge_attr: torch.Tensor,
                edge_attr2: torch.Tensor,
                edge_index: torch.LongTensor,
                edge_index2: torch.LongTensor,
                num_nodes: int) -> torch.Tensor:
        return scatter(
            edge_attr, edge_index[0], dim=0, dim_size=num_nodes, reduce='sum'
        ) + scatter(
            edge_attr2, edge_index2[0], dim=0, dim_size=num_nodes, reduce='sum'
        )

class GraphLevelPooling(nn.Module):
    """
    Define a graph-level pooling function for 2-DRFWL(2):

    W(G) = \sum_{d(u, v) <= 2} W(u, v)
    """
    def __init__(self):
        super().__init__()

    def forward(self, edge_attr: torch.Tensor,
                edge_attr2: torch.Tensor) -> torch.Tensor:
        return torch.sum(edge_attr, dim=0) + torch.sum(edge_attr2, dim=0)
