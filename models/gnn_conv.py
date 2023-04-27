"""
GNN layers
"""

import torch
import torch.nn as nn
from torch_scatter import scatter
import torch.nn.functional as F
from .mlp import MLP
from .norms import Normalization
from typing import Tuple, List, Dict
from torch_geometric.nn import global_mean_pool


class DR2FWL2Conv(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 norm_type: str,
                 eps: float,
                 train_eps: bool,
                 relu_last: bool,
                 add_root: bool = False):
        super().__init__()
        self.aggr_list: List[Tuple[int, int, int]] = []
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_type = norm_type
        self.relu_last = relu_last
        self.add_root = add_root
        self.inner_lin = nn.Linear(self.in_channels, self.out_channels)

        self.mlps = nn.ModuleDict()
        self.lins = nn.ModuleDict()
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps], requires_grad=True))
        else:
            self.register_buffer('eps', torch.tensor([eps]))

        if self.add_root:
            self.root_lin = nn.Linear(self.in_channels, self.out_channels)


    def reset_parameters(self):
        self.inner_lin.reset_parameters()
        for lin in self.lins.values():
            lin.reset_parameters()
        for lin in self.mlps.values():
            lin.reset_parameters()
        if self.add_root:
            self.root_lin.reset_parameters()


    def forward(self,
                edge_attrs: List[torch.Tensor],
                edge_indices: List[torch.LongTensor],
                triangles: Dict[Tuple[int, int, int], torch.LongTensor],
                inverse_edges: List[torch.LongTensor],
                batch: torch.LongTensor = None) -> List[torch.Tensor]:
        nums = [edge_attr.shape[0] for edge_attr in edge_attrs]
        aggr_out = [torch.tensor([0], device=edge_attrs[0].device) for _ in edge_attrs]

        edge_inner_attrs = [self.inner_lin(edge_attr) for edge_attr in edge_attrs]

        for aggr in self.aggr_list:
            aggr_c = self.parse_aggr(aggr)
            if aggr_c == 0:
                j = aggr[1]

                # 0j->j and j0->j aggregations
                s, e = edge_indices[j - 1]
                aggr_out[j] = aggr_out[j] + self.lins[str((j, j))](
                    F.relu(edge_inner_attrs[0][s] + edge_inner_attrs[0][e]))
                # jj->0 aggregation
                aggr_out[0] = aggr_out[0] + self.lins[str((j, 0))](scatter(F.relu(
                                    edge_inner_attrs[j]), s, dim=0, dim_size=nums[0]))
            elif aggr_c == 1:
                j = aggr[0]
                # jj->j aggregation
                tri = triangles[aggr]
                aggr_out[j] = aggr_out[j] + self.lins[str((j, j))](scatter(F.relu(
                    edge_inner_attrs[j][tri[1]] + edge_inner_attrs[j][tri[2]]), tri[0], dim=0, dim_size=nums[j]))
            elif aggr_c == 2:
                i, j = aggr[0], aggr[2]
                # ii->j aggregation
                tri = triangles[aggr]
                aggr_out[j] = aggr_out[j] + self.lins[str((i, i))](scatter(F.relu(
                    edge_inner_attrs[i][tri[0]] + edge_inner_attrs[i][tri[1]]), tri[2], dim=0, dim_size=nums[j]))
                # ij->i and ji->i aggregations
                out_iji = scatter(F.relu(edge_inner_attrs[i][tri[1]] +
                                         edge_inner_attrs[j][tri[2]]), tri[0], dim=0, dim_size=nums[i])
                aggr_out[i] = aggr_out[i] + self.lins[str((j, i))](out_iji + out_iji[inverse_edges[i - 1]])
            elif aggr_c == 3:
                i, j, k = aggr
                tri = triangles[aggr]

                out_jki = scatter(F.relu(edge_inner_attrs[j][tri[1]] +
                                         edge_inner_attrs[k][tri[2]]), tri[0], dim=0, dim_size=nums[i])
                aggr_out[i] = aggr_out[i] + self.lins[str((j, k))](out_jki + out_jki[inverse_edges[i - 1]])

                out_ikj = scatter(F.relu(edge_inner_attrs[i][tri[0]] +
                                         edge_inner_attrs[k][tri[2]]), tri[1], dim=0, dim_size=nums[j])

                aggr_out[j] = aggr_out[j] + self.lins[str((k, i))](out_ikj + out_ikj[inverse_edges[j - 1]])

                out_kij = scatter(F.relu(edge_inner_attrs[i][tri[0]] +
                                  edge_inner_attrs[j][tri[1]]), tri[2], dim=0, dim_size=nums[k])
                aggr_out[k] = aggr_out[k] + self.lins[str((i, j))](out_kij + out_kij[inverse_edges[k - 1]])


        if self.add_root:
            h_root = global_mean_pool(edge_attrs[0], batch)[batch] +  edge_attrs[0]
            aggr_out = [aggr_out[j] + F.tanh(self.root_lin(edge_attrs[j] + h_root)) if j == 0 else
                          aggr_out[j] + F.tanh(self.root_lin(edge_attrs[j] + h_root[edge_indices[j - 1][1]]))
                          for j in range(len(edge_attrs))]

        return [self.mlps[str(i)](edge_attrs[i] * (1 + self.eps) + aggr_out[i]) + edge_attrs[i]
                if str(i) in self.mlps.keys() else edge_attrs[i]
                for i in range(len(nums))] if not self.relu_last else [
            F.relu(self.mlps[str(i)](edge_attrs[i] * (1 + self.eps) + aggr_out[i])) + edge_attrs[i]
            if str(i) in self.mlps.keys() else edge_attrs[i]
            for i in range(len(nums))]

    def add_aggr(self, aggr_type: Tuple[int, int, int]):
        aggr_c = self.parse_aggr(aggr_type)

        self.aggr_list.append(aggr_type)
        # root node
        if aggr_c == 0:
            j = aggr_type[1]
            self.lins[str((j, j))] = nn.Linear(self.in_channels, self.in_channels)
            self.lins[str((j, 0))] = nn.Linear(self.in_channels, self.in_channels)
        elif aggr_c == 1:
            j = aggr_type[0]
            self.lins[str((j, j))] = nn.Linear(self.in_channels, self.in_channels)

        elif aggr_c == 2:
            i, j = aggr_type[0], aggr_type[2]
            self.lins[str((i, i))] = nn.Linear(self.in_channels, self.in_channels)
            self.lins[str((j, i))] = nn.Linear(self.in_channels, self.in_channels)
        elif aggr_c == 3:
            i, j, k = aggr_type
            self.lins[str((j, k))] = nn.Linear(self.in_channels, self.in_channels)
            self.lins[str((k, i))] = nn.Linear(self.in_channels, self.in_channels)
            self.lins[str((i, j))] = nn.Linear(self.in_channels, self.in_channels)

        for i in aggr_type:
            self.mlps[str(i)] = MLP(self.in_channels, self.out_channels, self.norm_type)

    def parse_aggr(self, aggr_type: Tuple[int, int, int]):
        i, j, k = aggr_type
        if i == 0:
            return 0
        elif i == k:
            return 1
        elif i == j:
            return 2
        elif i < j and j < k:
            return 3
        else:
            raise ValueError("Incorrect tuple order!")



"""
class DR2FWL2ConvNew(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 norm_type: str,
                 eps: float,
                 train_eps: bool,
                 relu_last: bool):
        super().__init__()
        self.aggr_list: List[Tuple[int, int, int]] = []
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_type = norm_type
        self.relu_last = relu_last
        self.inner_lin = nn.Linear(self.in_channels, self.out_channels)
        self.outer_mlp = MLP(self.in_channels, self.out_channels, "none")


        self.mlps = nn.ModuleDict()
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps], requires_grad=True))
        else:
            self.register_buffer('eps', torch.tensor([eps]))

    def reset_parameters(self):
        self.inner_lin.reset_parameters()
        self.outer_mlp.reset_parameters()

        for lin in self.mlps.values():
            lin.reset_parameters()


    def forward(self,
                edge_attrs: List[torch.Tensor],
                edge_indices: List[torch.LongTensor],
                triangles: Dict[Tuple[int, int, int], torch.LongTensor],
                inverse_edges: List[torch.LongTensor]) -> List[torch.Tensor]:
        nums = [edge_attr.shape[0] for edge_attr in edge_attrs]
        aggr_out = [torch.tensor([0], device=edge_attrs[0].device) for _ in edge_attrs]

        edge_inner_attrs = [self.inner_lin(edge_attr) for edge_attr in edge_attrs]

        for aggr in self.aggr_list:
            aggr_c = self.parse_aggr(aggr)
            if aggr_c == 0:
                j = aggr[1]

                # 0j->j and j0->j aggregations
                s, e = edge_indices[j - 1]
                aggr_out[j] = aggr_out[j] + self.outer_mlp(
                    F.relu(edge_inner_attrs[0][s] + edge_inner_attrs[0][e]))
                # jj->0 aggregation
                aggr_out[0] = aggr_out[0] + self.outer_mlp(scatter(F.relu(
                                    edge_inner_attrs[j]), s, dim=0, dim_size=nums[0]))
            elif aggr_c == 1:
                j = aggr[0]
                # jj->j aggregation
                tri = triangles[aggr]
                aggr_out[j] = aggr_out[j] + self.outer_mlp(scatter(F.relu(
                    edge_inner_attrs[j][tri[1]] + edge_inner_attrs[j][tri[2]]), tri[0], dim=0, dim_size=nums[j]))
            elif aggr_c == 2:
                i, j = aggr[0], aggr[2]
                # ii->j aggregation
                tri = triangles[aggr]
                aggr_out[j] = aggr_out[j] + self.outer_mlp(scatter(F.relu(
                    edge_inner_attrs[i][tri[0]] + edge_inner_attrs[i][tri[1]]), tri[2], dim=0, dim_size=nums[j]))
                # ij->i and ji->i aggregations
                out_iji = scatter(F.relu(edge_inner_attrs[i][tri[1]] +
                                         edge_inner_attrs[j][tri[2]]), tri[0], dim=0, dim_size=nums[i])
                aggr_out[i] = aggr_out[i] + self.outer_mlp(out_iji + out_iji[inverse_edges[i - 1]])
            elif aggr_c == 3:
                i, j, k = aggr
                tri = triangles[aggr]

                out_jki = scatter(F.relu(edge_inner_attrs[j][tri[1]] +
                                         edge_inner_attrs[k][tri[2]]), tri[0], dim=0, dim_size=nums[i])
                aggr_out[i] = aggr_out[i] + self.outer_mlp(out_jki + out_jki[inverse_edges[i - 1]])

                out_ikj = scatter(F.relu(edge_inner_attrs[i][tri[0]] +
                                         edge_inner_attrs[k][tri[2]]), tri[1], dim=0, dim_size=nums[j])

                aggr_out[j] = aggr_out[j] + self.outer_mlp(out_ikj + out_ikj[inverse_edges[j - 1]])

                out_kij = scatter(F.relu(edge_inner_attrs[i][tri[0]] +
                                  edge_inner_attrs[j][tri[1]]), tri[2], dim=0, dim_size=nums[k])
                aggr_out[k] = aggr_out[k] + self.outer_mlp(out_kij + out_kij[inverse_edges[k - 1]])

        return [self.mlps[str(i)](edge_attrs[i] * (1 + self.eps) + aggr_out[i]) + edge_attrs[i]
                if str(i) in self.mlps.keys() else edge_attrs[i]
                for i in range(len(nums))] if not self.relu_last else [
            F.relu(self.mlps[str(i)](edge_attrs[i] * (1 + self.eps) + aggr_out[i])) + edge_attrs[i]
            if str(i) in self.mlps.keys() else edge_attrs[i]
            for i in range(len(nums))]

    def add_aggr(self, aggr_type: Tuple[int, int, int]):
        aggr_c = self.parse_aggr(aggr_type)
        self.aggr_list.append(aggr_type)
        for i in aggr_type:
            self.mlps[str(i)] = MLP(self.in_channels, self.out_channels, self.norm_type)

    def parse_aggr(self, aggr_type: Tuple[int, int, int]):
        i, j, k = aggr_type
        if i == 0:
            return 0
        elif i == k:
            return 1
        elif i == j:
            return 2
        elif i < j and j < k:
            return 3
        else:
            raise ValueError("Incorrect tuple order!")

"""

class DR2FWL2ConvSimple(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 norm_type: str,
                 eps: float,
                 train_eps: bool,
                 relu_last: bool,
                 add_graph: bool = False):
        super().__init__()
        self.aggr_list: List[Tuple[int, int, int]] = []
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_type = norm_type
        self.initial_eps = eps
        self.relu_last = relu_last
        self.add_graph = add_graph

        # combine inner linear into one
        self.inner_lin = nn.Linear(self.in_channels, self.in_channels)
        self.mlps = nn.ModuleDict()
        self.lins = nn.ModuleDict()
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([self.initial_eps], requires_grad=True))
        else:
            self.register_buffer('eps', torch.tensor([self.initial_eps]))

        if self.add_graph:
            self.graph_proj = nn.Linear(self.in_channels, self.out_channels)
            self.root_proj = nn.Linear(self.out_channels, self.out_channels)
            if train_eps:
                self.graph_eps = nn.Parameter(torch.tensor([self.initial_eps], requires_grad=True))
                self.root_eps = nn.Parameter(torch.tensor([self.initial_eps], requires_grad=True))

            else:
                self.register_buffer('graph_eps', torch.tensor([self.initial_eps]))
                self.register_buffer('root_eps', torch.tensor([self.initial_eps]))


    def reset_parameters(self):
        self.inner_lin.reset_parameters()

        for lin in self.lins.values():
            lin.reset_parameters()
        for lin in self.mlps.values():
            lin.reset_parameters()

        if self.add_graph:
            self.graph_proj.reset_parameters()
            self.root_proj.reset_parameters()
            self.graph_eps.data.fill_(self.initial_eps)
            self.root_eps.data.fill_(self.initial_eps)



    def forward(self, 
                edge_attrs: List[torch.Tensor],
                edge_indices: List[torch.LongTensor],
                triangles: Dict[Tuple[int, int, int], torch.LongTensor],
                inverse_edges: List[torch.LongTensor],
                batch: torch.LongTensor = None) -> List[torch.Tensor]:
        nums = [edge_attr.shape[0] for edge_attr in edge_attrs]
        aggr_out = [torch.tensor([0], device=edge_attrs[0].device) for _ in edge_attrs]

        for aggr in self.aggr_list:
            aggr_c = self.parse_aggr(aggr)

            if aggr_c == 0:
                j = aggr[1]
                # 0j->j and j0->j aggregations
                s, e = edge_indices[j - 1]
                aggr_out[j] = aggr_out[j] + ( # for 0-hop no outer lin
                (F.relu(
                    self.inner_lin(edge_attrs[0][s] + edge_attrs[0][e]))))
                # jj->0 aggregation
                aggr_out[0] = aggr_out[0] + ( # for 0-hop no outer lin
                (scatter(F.relu(
                    self.inner_lin(edge_attrs[j])), s, dim=0, dim_size=nums[0])))
            elif aggr_c == 1:
                j = aggr[0]
                # jj->j aggregation
                tri = triangles[aggr]
                aggr_out[j] = aggr_out[j] + self.lins[str((j, j, j))] (scatter(F.relu(
                    self.inner_lin(edge_attrs[j][tri[1]]+edge_attrs[j][tri[2]])),
                    tri[0], dim=0, dim_size=nums[j]
                ))
            elif aggr_c == 2:
                i, j = aggr[0], aggr[2]
                # ii->j aggregation
                tri = triangles[aggr]
                aggr_out[j] = aggr_out[j] + self.lins[str((j, i, i))] (scatter(F.relu(
                    self.inner_lin(edge_attrs[i][tri[0]]+edge_attrs[i][tri[1]])),
                    tri[2], dim=0, dim_size=nums[j]
                ))
                # ij->i and ji->i aggregations
                out_iji = scatter(F.relu(
                    self.inner_lin(edge_attrs[i][tri[1]]+edge_attrs[j][tri[2]])),
                    tri[0], dim=0, dim_size=nums[i]
                )
                aggr_out[i] = aggr_out[i] + self.lins[str((j, i, i))] (out_iji + out_iji[inverse_edges[i - 1]])
            elif aggr_c == 3:
                i, j, k = aggr
                tri = triangles[aggr]
                
                out_jki = scatter(F.relu(
                    self.inner_lin
                    (edge_attrs[j][tri[1]]+edge_attrs[k][tri[2]])),
                    tri[0], dim=0, dim_size=nums[i]
                )
                aggr_out[i] = aggr_out[i] + self.lins[str((i, j, k))] (out_jki + out_jki[inverse_edges[i - 1]])
                
                out_ikj = scatter(F.relu(
                    self.inner_lin
                    (edge_attrs[i][tri[0]]+edge_attrs[k][tri[2]])),
                    tri[1], dim=0, dim_size=nums[j]
                )
                aggr_out[j] = aggr_out[j] + self.lins[str((i, j, k))] (out_ikj + out_ikj[inverse_edges[j - 1]])

                out_kij = scatter(F.relu(
                    self.inner_lin
                    (edge_attrs[i][tri[0]]+edge_attrs[j][tri[1]])),
                    tri[2], dim=0, dim_size=nums[k]
                )
                aggr_out[k] = aggr_out[k] + self.lins[str((i, j, k))] (out_kij + out_kij[inverse_edges[k - 1]])

        if self.add_graph:
            h_graph = global_mean_pool(edge_attrs[0], batch)[batch]
            h_root = F.elu(self.graph_proj((1 + self.graph_eps) * edge_attrs[0] + h_graph))
            aggr_root = [F.tanh(self.root_proj((1 + self.root_eps) * edge_attrs[j] + h_root)) if j == 0 else
                         F.tanh(self.root_proj((1 + self.root_eps) * edge_attrs[j] + h_root[edge_indices[j-1][1]]))
                          for j in range(len(nums))]

            aggr_out = [aggr_out[j] + aggr_root[j] for j in range(len(nums))]

        
        return [self.mlps[str(i)](edge_attrs[i] * (1 + self.eps) + aggr_out[i]) + edge_attrs[i]
                if str(i) in self.mlps.keys() else edge_attrs[i]
                for i in range(len(nums))] if not self.relu_last else [
                F.relu(self.mlps[str(i)](edge_attrs[i] * (1 + self.eps) + aggr_out[i])) + edge_attrs[i]
                if str(i) in self.mlps.keys() else edge_attrs[i]
                    for i in range(len(nums))]

    def add_aggr(self, aggr_type: Tuple[int, int, int]):
        aggr_c = self.parse_aggr(aggr_type)

        self.aggr_list.append(aggr_type)

        if aggr_c == 1:
            j = aggr_type[0]
            self.lins[str((j, j, j))] = nn.Linear(self.in_channels, self.in_channels)
        elif aggr_c == 2:
            i, j = aggr_type[0], aggr_type[2]
            self.lins[str((j, i, i))] = nn.Linear(self.in_channels, self.in_channels)
        elif aggr_c == 3:
            i, j, k = aggr_type
            self.lins[str((i, j, k))] = nn.Linear(self.in_channels, self.in_channels)

        for i in aggr_type:
            self.mlps[str(i)] = MLP(self.in_channels, self.out_channels, self.norm_type)

    def parse_aggr(self, aggr_type: Tuple[int, int, int]):
        i, j, k = aggr_type
        if i == 0:
            return 0
        elif i == k:
            return 1
        elif i == j:
            return 2
        elif i < j and j < k:
            return 3
        else:
            raise ValueError("Incorrect tuple order!")
