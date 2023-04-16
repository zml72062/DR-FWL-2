import torch
import torch.nn as nn
from torch_scatter import scatter
from pygmmpp.nn.model import MLP

class GraphLevelPooling(nn.Module):
    r"""Graph level pooling module for DR2FWL.
    """
    def __init__(self):
        super().__init__()
    def forward(self,
                edge_attr0: torch.Tensor,
                edge_attr1: torch.Tensor,
                edge_attr2: torch.Tensor,
                edge_index: torch.LongTensor,
                edge_index2: torch.LongTensor,
                num_nodes: int,
                batch) -> torch.Tensor:
        node_emb = edge_attr0 + scatter(edge_attr1, edge_index[0], dim=0, dim_size=num_nodes, reduce='sum') + \
                   scatter(edge_attr2, edge_index2[0], dim=0, dim_size=num_nodes, reduce='sum')
        return scatter(node_emb, batch, dim=0, reduce="mean")




class NodeLevelPooling(nn.Module):
    r"""Node level pooling module for DR2FWL.
    """
    def __init__(self):
        super().__init__()

    def forward(self,
                edge_attr0: torch.Tensor,
                edge_attr1: torch.Tensor,
                edge_attr2: torch.Tensor,
                edge_index: torch.LongTensor,
                edge_index2: torch.LongTensor,
                num_nodes: int) -> torch.Tensor:
        return edge_attr0 + scatter(
            edge_attr1, edge_index[0], dim=0, dim_size=num_nodes, reduce='sum'
        ) + scatter(
            edge_attr2, edge_index2[0], dim=0, dim_size=num_nodes, reduce='sum'
        )