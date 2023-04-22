import torch

from .gnn_conv_count import *
from .utils import clones
from .norms import Normalization
from .auxiliaries import MultipleComponentLinear

class DR2FWL2Kernel(torch.nn.Module):
    def __init__(self,
                 hidden_channels: int,
                 num_layers: int,
                 eps: float = 0.0,
                 train_eps: bool = False,
                 norm_type: str = "batch_norm",
                 norm_between_layers: str = "batch_norm",
                 residual: str = "last",
                 drop_prob: float = 0.0):

        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.initial_eps = eps
        self.train_eps = train_eps
        self.norm_type = norm_type
        self.residual = residual
        self.drop_prob = drop_prob

        gnn = DR2FWL2Conv(hidden_channels,
                          hidden_channels,
                          norm_type,
                          eps,
                          train_eps, False)
        norm = Normalization(hidden_channels, norm_between_layers)

        self.gnns = clones(gnn, num_layers)
        self.norms = clones(norm, num_layers)

        if self.residual != 'cat':
            self.out_lin = MultipleComponentLinear(self.hidden_channels,
                                                   self.hidden_channels,
                                                   self.drop_prob)

        else:
            self.out_lin = MultipleComponentLinear(self.hidden_channels * (1 + self.num_layers),
                                                   self.hidden_channels,
                                                   self.drop_prob)

        self.reset_parameters()
    
    def add_aggr(self, i: int, j: int, k: int):
        for g in self.gnns:
            g.add_aggr((i, j, k))

    def reset_parameters(self):
        for g in self.gnns:
            g.reset_parameters()
        for g in self.norms:
            g.reset_parameters()

        self.out_lin.reset_parameters()


    def forward(self,
                edge_attrs: List[torch.Tensor],
                edge_indices: List[torch.LongTensor],
                triangles: Dict[Tuple[int, int, int], torch.LongTensor],
                inverse_edges: List[torch.LongTensor]) \
            -> List[torch.Tensor]:

        emb_list = edge_attrs

        for i in range(self.num_layers):
            edge_attrs = self.gnns[i](edge_attrs,
                                      edge_indices,
                                      triangles,
                                      inverse_edges)
            edge_attrs = [F.relu(self.norms[i](e)) for e in edge_attrs]
            edge_attrs = [F.dropout(e, p=self.drop_prob, training=self.training)
                          for e in edge_attrs]
            if self.residual == 'last':
                emb_list = edge_attrs
            elif self.residual == 'add':
                emb_list = [e + f for (e, f) in zip(edge_attrs, emb_list)]
            elif self.residual == 'cat':
                emb_list = [torch.cat([e, f], dim=-1) for (e, f) in zip(edge_attrs, emb_list)]
        
        return self.out_lin(emb_list)
