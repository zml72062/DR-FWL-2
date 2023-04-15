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
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self,
                num_edges: int,
                x_edge : torch.Tensor,
                edge_index_ij: torch.LongTensor,
                ) -> torch.Tensor:
        #x_ik = x_ik[edge_index_ik]
        #x_kj = x_kj[edge_index_kj]
        #x_edge = F.relu(x_ik + x_kj)
        out = scatter(x_edge, edge_index_ij, dim=0, dim_size=num_edges, reduce="sum")
        return out

class DR2FWL2Conv(nn.Module):
    r"""A new gnn conv for DRFWL2.
    Args:
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 add_112: bool = True,
                 add_212: bool = True,
                 add_222: bool = True,
                 add_vv: bool = False,
                 eps: float = 0.,
                 norm_type: str = "batch_norm"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_112 = add_112
        self.add_212 = add_212
        self.add_222 = add_222
        self.add_vv = add_vv
        self.initial_eps = eps
        self.norm_type = norm_type

        set_agg = MultisetAggregation(self.in_channels, self.out_channels)
        mlp = MLP(self.in_channels, self.out_channels, self.norm_type)
        proj = nn.Linear(self.in_channels, self.out_channels)
        norm = Normalization(self.out_channels, self.norm_type)
        eps = torch.nn.Parameter(torch.Tensor([self.initial_eps]))

        self.agg011 = c(set_agg)
        self.mlp0 = c(mlp)
        self.proj0 = c(proj)
        self.norm0 = c(norm)
        self.eps0 = c(eps)

        self.agg111 = c(set_agg)
        self.mlp1 = c(mlp)
        self.proj1 = c(proj)
        self.norm1 = c(norm)
        self.eps1 = c(eps)


        if self.add_112 or self.add_212 or self.add_222:
            self.mlp2 = c(mlp)
            self.proj2 = c(proj)
            self.norm2 = c(norm)
            self.eps2 = c(eps)

        if self.add_112:
            self.agg112 = c(set_agg)
            self.agg211 = c(set_agg)

        if self.add_212:
            self.agg122 = c(set_agg)
            self.agg212 = c(set_agg)

        if self.add_222:
            self.agg222 = c(set_agg)

        if self.add_vv:
            self.mlpvv = c(mlp)
            self.normvv = c(norm)
            self.epsvv = c(eps)

        self.reset_parameters()

    def reset_parameters(self):
        self.agg011.reset_parameters()
        self.mlp0.reset_parameters()
        self.proj0.reset_parameters()
        self.norm0.reset_parameters()
        self.eps0.data.fill_(self.initial_eps)

        self.agg111.reset_parameters()
        self.mlp1.reset_parameters()
        self.proj1.reset_parameters()
        self.norm1.reset_parameters()
        self.eps1.data.fill_(self.initial_eps)


        if self.add_112 or self.add_212 or self.add_222:
            self.mlp2.reset_parameters()
            self.proj2.reset_parameters()
            self.norm2.reset_parameters()
            self.eps2.data.fill_(self.initial_eps)

        if self.add_112:
            self.agg112.reset_parameters()
            self.agg211.reset_parameters()

        if self.add_212:
            self.agg212.reset_parameters()
            self.agg122.reset_parameters()

        if self.add_222:
            self.agg222.reset_parameters()

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

        # edge attr 0
        x_011_edge = F.relu(self.proj0(edge_attr1[ik011] + edge_attr1[ik011]))
        edge_attr0_out += self.norm0(self.mlp0((1 + self.eps0) * edge_attr0 +
                                     self.agg011(num_edge0, x_011_edge, ij011)))

        x_111_edge = F.relu(self.proj1(edge_attr1[ik111] + edge_attr1[kj111]))
        edge_attr1_out += self.agg111(num_edge1, x_111_edge, ij111)

        if self.add_112:
            x_112_edge = F.relu(self.proj1(edge_attr1[ik112] + edge_attr2[kj112]))
            edge_attr_112 = self.agg112(num_edge1, x_112_edge, ij112)
            edge_attr1_out += edge_attr_112 + edge_attr_112[inverse_edge_1]
            x_211_edge = F.relu(self.proj2(edge_attr1[ij112] + edge_attr1[ik112]))
            edge_attr2_out += self.agg211(num_edge2, x_211_edge, kj112)

        if self.add_212:
            x_122_edge = F.relu(self.proj1(edge_attr2[ik122] + edge_attr2[kj122]))
            edge_attr1_out += self.agg122(num_edge1, x_122_edge,  ij122)

            x_212_edge = F.relu(self.proj2(edge_attr1[ij122] + edge_attr2[kj122]))
            edge_attr_212 = self.agg212(num_edge2, x_212_edge, ik122)
            edge_attr2_out += edge_attr_212 + edge_attr_212[inverse_edge_2]

        if self.add_222:
            x_222_edge = F.relu(self.proj2(edge_attr2[ik222] + edge_attr2[kj222]))
            edge_attr2_out += self.agg222(num_edge2, x_222_edge, ij222)

        edge_attr1_out = self.norm1(self.mlp1((1 + self.eps1) * edge_attr1 + edge_attr1_out))
        edge_attr2_out = self.norm2(self.mlp2((1 + self.eps2) * edge_attr2 + edge_attr2_out))

        if self.add_vv:
            vv_out = self.normvv(self.mlpvv((1 + self.epsvv) *
                                    torch.cat([edge_attr0, edge_attr1, edge_attr2], dim=0)
                                    + edge_attr0[torch.cat([edge_index0[1], edge_index[1], edge_index2[1]], dim=-1)]))

            edge_attr0_out += vv_out[:num_edge0]
            edge_attr1_out += vv_out[num_edge0: num_edge1 + num_edge0]
            edge_attr2_out += vv_out[num_edge1 + num_edge0:]

        return edge_attr0_out, edge_attr1_out, edge_attr2_out
