import torch
import torch.nn as nn

from torch_scatter import scatter
from pygmmpp.nn.model import MLP
from typing import Optional

class DR2FWL2Conv(nn.Module):
    """
    TODO: Add formula for this.
    """
    def __init__(self, in_channels: int,
                 hidden_channels: int,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 eps: float = 0.0,
                 eps2: float = 0.0,
                 norm: Optional[str] = None):
        super().__init__()
        self.mlps = nn.ModuleList()
        self.eps = eps
        self.eps2 = eps2
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

    def forward(self,
                edge_attr: torch.Tensor,
                edge_attr2: torch.Tensor,
                triangle_1_1_1: torch.LongTensor,
                triangle_1_1_2: torch.LongTensor,
                triangle_1_2_2: torch.LongTensor,
                triangle_2_2_2: torch.LongTensor,
                inverse_edge_1: torch.LongTensor,
                inverse_edge_2: torch.LongTensor):
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
    
class DR2FWL2(nn.Module):
    # TODO: implement this
    def __init__(self,):
        pass
    
    def forward(self,):
        pass

if __name__ == '__main__':
    from pygmmpp.utils import compose
    from preprocess import (generate_k_hop_neighbor,
                            generate_k_hop_neighbor_feature,
                            ngnn_transform,
                            generate_lkm_triangle,
                            generate_inverse_edge)
    from pygmmpp.datasets import QM9
    from pygmmpp.data import DataLoader, Data
    from torch_geometric.utils.undirected import is_undirected
    from batch import collate

    dataset = QM9(root='data/test/QM9',
                        pre_transform=compose(
        [generate_k_hop_neighbor(2), 
        generate_k_hop_neighbor_feature(2),
        generate_lkm_triangle(1, 1, 1),
        generate_lkm_triangle(1, 1, 2),
        generate_lkm_triangle(1, 2, 2),
        generate_lkm_triangle(2, 2, 2),
        generate_inverse_edge(2)
        ])
    )
    loader = DataLoader(dataset, batch_size=1, collator=collate)
    for batch in loader:
        assert is_undirected(batch.edge_index, batch.edge_attr, batch.num_nodes)
        batch.edge_attr = nn.Linear(15, 32)(batch.edge_attr)
        batch.edge_attr2 = nn.Linear(11, 32)(batch.edge_attr2)
        assert is_undirected(batch.edge_index, batch.edge_attr, batch.num_nodes)
        DR2FWL2Conv(32, 64, 1, norm='batch_norm')(
            batch.edge_attr,
            batch.edge_attr2,
            batch.triangle_1_1_1,
            batch.triangle_1_1_2,
            batch.triangle_1_2_2,
            batch.triangle_2_2_2,
            batch.inverse_edge_1,
            batch.inverse_edge_2
        )
        assert is_undirected(batch.edge_index, batch.edge_attr, batch.num_nodes)

