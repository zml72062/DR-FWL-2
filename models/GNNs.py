
from .gnn_conv import *
from .utils import clones
from .mlp import MLP
from .auxiliaries import MultipleComponentLinear



class DR2FWL2Kernel(torch.nn.Module):
    """
    Define a 2-Distance Restricted FWL(2) GNN kernel by stacking `DR2FWL2Conv`
    layers. This kernel can be further combined with node-level/graph-level
    pooling layers as it is applied on different tasks.
    """
    def __init__(self,
                 hidden_channels: int,
                 num_layers: int,
                 add_112: bool = True,
                 add_212: bool = True,
                 add_222: bool = True,
                 add_vv: bool = False,
                 eps: float = 0.,
                 norm_type: str = "batch_norm",
                 residual: str = "none",
                 drop_prob: float = 0.0):

        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.add_112 = add_112
        self.add_212 = add_212
        self.add_222 = add_222
        self.add_vv = add_vv
        self.initial_eps = eps
        self.norm_type = norm_type
        self.residual = residual
        self.drop_prob = drop_prob


        gnn = DR2FWL2Conv(self.hidden_channels,
                          self.hidden_channels,
                          self.add_112,
                          self.add_212,
                          self.add_222,
                          self.add_vv,
                          self.initial_eps,
                          self.norm_type)

        self.gnns = clones(gnn, num_layers)

        if self.residual != 'cat':
            self.out_lin = MultipleComponentLinear(self.hidden_channels,
                                                   self.hidden_channels,
                                                   self.drop_prob)

        else:
            self.out_lin = MultipleComponentLinear(self.hidden_channels * (1 + self.num_layers),
                                                   self.hidden_channels,
                                                   self.drop_prob)

        self.reset_parameters()

    def reset_parameters(self):
        for g in self.gnns:
            g.reset_parameters()

        self.out_lin.reset_parameters()


    def forward(self,
                edge_attr0: torch.Tensor,
                edge_attr1: torch.Tensor,
                edge_attr2: torch.Tensor,
                edge_index0: torch.Tensor,
                edge_index: torch.LongTensor,
                edge_index2: torch.LongTensor,
                triangle_0_1_1: torch.LongTensor,
                triangle_1_1_1: torch.LongTensor,
                triangle_1_1_2: torch.LongTensor,
                triangle_1_2_2: torch.LongTensor,
                triangle_2_2_2: torch.LongTensor,
                inverse_edge_1: torch.LongTensor,
                inverse_edge_2: torch.LongTensor,
                edge_emb_list: list = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:


        emb_list0 = [edge_attr0]
        emb_list1 = [edge_attr1]
        emb_list2 = [edge_attr2]

        for i in range(self.num_layers):
            edge_attr0 = emb_list0[i]
            edge_attr1 = emb_list1[i]
            edge_attr2 = emb_list2[i]
            if edge_emb_list is not None:
                edge_attr1 += edge_emb_list[i]

            edge_attr0_out, edge_attr1_out, edge_attr2_out = self.gnns[i](edge_attr0,
                                                                          edge_attr1,
                                                                          edge_attr2,
                                                                          edge_index0,
                                                                          edge_index,
                                                                          edge_index2,
                                                                          triangle_0_1_1,
                                                                          triangle_1_1_1,
                                                                          triangle_1_1_2,
                                                                          triangle_1_2_2,
                                                                          triangle_2_2_2,
                                                                          inverse_edge_1,
                                                                          inverse_edge_2)

            edge_attr0_out, edge_attr1_out, edge_attr2_out = F.dropout(
                edge_attr0_out, p=self.drop_prob, training=self.training), F.dropout(
                edge_attr1_out, p=self.drop_prob, training=self.training), F.dropout(
                edge_attr2_out, p=self.drop_prob, training=self.training)

            emb_list0.append(edge_attr0_out)
            emb_list1.append(edge_attr1_out)
            emb_list2.append(edge_attr2_out)

        if self.residual == "last":
            return self.out_lin((emb_list0[-1], emb_list1[-1], emb_list2[-1]))
        elif self.residual == 'add':
            return self.out_lin((sum(emb_list0), sum(emb_list1), sum(emb_list2)))
        elif self.residual == 'cat':
            return self.out_lin((torch.cat(emb_list0, dim=-1),
                                torch.cat(emb_list1, dim=-1),
                                torch.cat(emb_list2, dim=-1)))

