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
                edge_attr_list: list,
                edge_index_list: list,
                num_nodes: int,
                batch) -> torch.Tensor:
        list_len = len(edge_attr_list)
        node_emb = sum([scatter(edge_attr_list[i], edge_index_list[i][0], dim=0, dim_size=num_nodes, reduce='sum') for i in range(list_len)])
        return scatter(node_emb, batch, dim=0, reduce="mean")




class NodeLevelPooling(nn.Module):
    r"""Node level pooling module for DR2FWL.
    """
    def __init__(self):
        super().__init__()

    def forward(self,
                edge_attr_list: list,
                edge_index_list: list,
                num_nodes: int) -> torch.Tensor:
        list_len = len(edge_attr_list)
        node_emb = sum([scatter(edge_attr_list[i], edge_index_list[i][0], dim=0, dim_size=num_nodes, reduce='sum') for i in range(list_len)])
        return node_emb