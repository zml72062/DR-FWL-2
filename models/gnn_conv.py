"""
GNN layers
"""

import torch
import torch.nn as nn
from torch_scatter import scatter
import torch.nn.functional as F
from copy import deepcopy as c
from .mlp import MLP
from .norms import Normalization
from typing import Tuple

class MultisetAggregation(nn.Module):
    r"""Multiset aggregation module with elements of 2-sets.
    Args:
        in_channels (int): Input size.
        out_channels (int): Output size.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj = nn.Linear(self.in_channels * 2, self.out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.proj.reset_parameters()

    def forward(self,
                num_edges: int,
                x_ik: torch.Tensor,
                x_kj: torch.Tensor,
                edge_index_ij: torch.LongTensor,
                edge_index_ik: torch.LongTensor,
                edge_index_kj: torch.LongTensor
                ) -> torch.Tensor:

        x_edge = F.relu(self.proj(torch.cat([x_ik[edge_index_ik], x_kj[edge_index_kj]], dim=-1)))
        out = scatter(x_edge, edge_index_ij, dim=0, dim_size=num_edges, reduce="sum")
        return out

class DR2FWL2Conv(nn.Module):
    r"""Convolutional layer of distance restricted 2-FWL. The update formula is
        ..math::
        W(u, v) = \text{HASH}(W(u, v), ({{(W(w, v), W(u, w))|
                w \in \mathcal{N}_{i}(u) \cap \mathcal{N}_{j}(v)}})_{0 \leq i, j \leq 2 })
    Args:
        in_channels (int): input size.
        out_channels (int): output size.
        add_0 (bool): If true, add multiset aggregation involves root nodes.
        add_112 (bool): If true, add all multiset aggregations for triangle 1-1-2.
        add_212 (bool): If true, add all multiset aggregations for triangle 2-1-2.
        add_222 (bool): If true, add all multiset aggregations for triangle 2-2-2.
        add_vv (bool): If true, for each :math::W(u, v), add :math::W(v, v) as additional aggregation.
                        Only works if add_0 is true.
        eps (bool): Epsilon for distinguishing W(u, v) in aggregation, default is trainable.
        norm_type (str): Normalization type after each layer, choose from ("none", "batch_norm", "layer_norm").
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 add_0: bool = True,
                 add_112: bool = True,
                 add_212: bool = True,
                 add_222: bool = True,
                 add_vv: bool = False,
                 eps: float = 0.,
                 train_eps: bool = False,
                 norm_type: str = "batch_norm"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_0 = add_0
        self.add_112 = add_112
        self.add_212 = add_212
        self.add_222 = add_222
        self.add_vv = add_vv
        self.initial_eps = eps
        self.train_eps = train_eps
        self.norm_type = norm_type

        set_agg = MultisetAggregation(self.in_channels, self.out_channels)
        mlp = MLP(self.in_channels, self.out_channels, self.norm_type)
        norm = Normalization(self.out_channels, self.norm_type)
        if self.train_eps:
            eps = torch.nn.Parameter(torch.Tensor([self.initial_eps]))


        self.agg0 = c(set_agg)
        self.mlp0 = c(mlp)
        self.norm0 = c(norm)
        if self.train_eps:
            self.eps0 = c(eps)
        else:
            self.register_buffer('eps0', torch.Tensor([self.initial_eps]))

        self.agg1 = c(set_agg)
        self.mlp1 = c(mlp)
        self.norm1 = c(norm)
        if self.train_eps:
            self.eps1 = c(eps)
        else:
            self.register_buffer('eps1', torch.Tensor([self.initial_eps]))


        self.agg2 = c(set_agg)
        self.mlp2 = c(mlp)
        self.norm2 = c(norm)
        if self.train_eps:
            self.eps2 = c(eps)
        else:
            self.register_buffer('eps2', torch.Tensor([self.initial_eps]))

        if self.add_vv:
            self.mlpvv = c(mlp)
            self.normvv = c(norm)
            if self.train_eps:
                self.epsvv = c(eps)
            else:
                self.register_buffer('epsvv', torch.Tensor([self.initial_eps]))

        self.reset_parameters()

    def reset_parameters(self):
        self.agg0.reset_parameters()
        self.mlp0.reset_parameters()
        self.norm0.reset_parameters()
        self.eps0.data.fill_(self.initial_eps)

        self.agg1.reset_parameters()
        self.mlp1.reset_parameters()
        self.norm1.reset_parameters()
        self.eps1.data.fill_(self.initial_eps)


        self.agg2.reset_parameters()
        self.mlp2.reset_parameters()
        self.norm2.reset_parameters()
        self.eps2.data.fill_(self.initial_eps)


        if self.add_vv:
            self.mlpvv.reset_parameters()
            self.normvv.reset_parameters()
            self.epsvv.data.fill_(self.initial_eps)


    def forward(self,
                edge_attr0: torch.Tensor,
                edge_attr1: torch.Tensor,
                edge_attr2: torch.Tensor,
                edge_index0: torch.LongTensor,
                edge_index: torch.LongTensor,
                edge_index2: torch.LongTensor,
                triangle_0_1_1: torch.LongTensor,
                triangle_1_1_1: torch.LongTensor,
                triangle_1_1_2: torch.LongTensor,
                triangle_1_2_2: torch.LongTensor,
                triangle_2_2_2: torch.LongTensor,
                inverse_edge_1: torch.LongTensor,
                inverse_edge_2: torch.LongTensor,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # For undirected graphs, the direction of edges is unimportant
        ij011, ik011, kj011 = triangle_0_1_1
        ij111, ik111, kj111 = triangle_1_1_1
        ij112, ik112, kj112 = triangle_1_1_2
        ij122, ik122, kj122 = triangle_1_2_2
        ij222, ik222, kj222 = triangle_2_2_2

        num_edge0 = edge_attr0.size(0)
        num_edge1 = edge_attr1.size(0)
        num_edge2 = edge_attr2.size(0)

        edge_attr0_out = torch.zeros_like(edge_attr0)
        edge_attr1_out = torch.zeros_like(edge_attr1)
        edge_attr2_out = torch.zeros_like(edge_attr2)


        if self.add_0:
            #011
            edge_attr0_out += self.agg0(num_edge0, edge_attr1, edge_attr1, ij011, ik011, kj011)
            edge_attr0_out = self.norm0(self.mlp0((1 + self.eps0) * edge_attr0 + edge_attr0_out))

            #101
            edge_attr_101 = self.agg1(num_edge1, edge_attr0, edge_attr1, ik011, ij011, kj011)
            edge_attr1_out += (edge_attr_101 + edge_attr_101[inverse_edge_1]) / 2

        #111
        edge_attr_111 = self.agg1(num_edge1, edge_attr1, edge_attr1, ij111, ik111, kj111)
        edge_attr1_out += (edge_attr_111 + edge_attr_111[inverse_edge_1]) / 2

        if self.add_112:
            #112
            edge_attr_112 = self.agg1(num_edge1, edge_attr1, edge_attr2, ij112, ik112, kj112)
            edge_attr1_out += (edge_attr_112 + edge_attr_112[inverse_edge_1]) / 2
            #211
            edge_attr_211 = self.agg2(num_edge2, edge_attr1, edge_attr1, kj112, ij112, ik112)
            edge_attr2_out += (edge_attr_211 + edge_attr_211[inverse_edge_2]) / 2

        if self.add_212:
            #122
            edge_attr_122 = self.agg1(num_edge1, edge_attr2, edge_attr2, ij122, ik122, kj122)
            edge_attr1_out += (edge_attr_122 + edge_attr_122[inverse_edge_1]) / 2
            #212
            edge_attr_212 = self.agg2(num_edge2, edge_attr1, edge_attr2, ik122, ij122, kj122)
            edge_attr2_out += (edge_attr_212 + edge_attr_212[inverse_edge_2]) / 2

        if self.add_222:
            #222
            edge_attr_222 = self.agg2(num_edge2, edge_attr2, edge_attr2, ij222, ik222, kj222)
            edge_attr2_out += (edge_attr_222 + edge_attr_222[inverse_edge_2]) / 2

        edge_attr1_out = self.norm1(self.mlp1((1 + self.eps1) * edge_attr1 + edge_attr1_out))
        edge_attr2_out = self.norm2(self.mlp2((1 + self.eps2) * edge_attr2 + edge_attr2_out))

        if self.add_vv and self.add_0:
            vv_out = self.normvv(self.mlpvv((1 + self.epsvv) *
                                    torch.cat([edge_attr0, edge_attr1, edge_attr2], dim=0)
                                    + edge_attr0[torch.cat([edge_index0[1], edge_index[1], edge_index2[1]], dim=-1)]))

            edge_attr0_out += vv_out[:num_edge0]
            edge_attr1_out += vv_out[num_edge0: num_edge1 + num_edge0]
            edge_attr2_out += vv_out[num_edge1 + num_edge0:]

        return edge_attr0_out, edge_attr1_out, edge_attr2_out
