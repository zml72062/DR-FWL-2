import torch
import torch.nn as nn
from torch_scatter import scatter
import torch.nn.functional as F
class GraphLevelPooling(nn.Module):
    r"""Graph level pooling module for DR2FWL.
    """
    def __init__(self,
                 hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.lin = nn.Linear(self.hidden_channels, self.hidden_channels)
    def forward(self,
                edge_attr_list: list,
                edge_index_list: list,
                num_nodes: int,
                batch) -> torch.Tensor:
        list_len = len(edge_attr_list)
        node_emb = sum([scatter(edge_attr_list[i], edge_index_list[i-1][0], dim=0, dim_size=num_nodes, reduce='sum') for i in range(1, list_len)])
        node_emb = F.elu(self.lin(node_emb + edge_attr_list[0]))
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
        node_emb = sum([scatter(edge_attr_list[i], edge_index_list[i-1][0], dim=0, dim_size=num_nodes, reduce='sum') for i in range(1, list_len)])
        return node_emb + edge_attr_list[0]