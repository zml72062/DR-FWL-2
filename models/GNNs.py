import torch

from .gnn_conv import *
from .utils import clones
from .mlp import MLP
from .auxiliaries import MultipleComponentLinear



class DR2FWL2Kernel(torch.nn.Module):
    r"""Define a 2-Distance Restricted FWL(2) GNN kernel by stacking `DR2FWL2Conv`
        layers. This kernel can be further combined with node-level/graph-level
        pooling layers as it is applied on different tasks.
    Args:
        hidden_channels (int): Hidden size of the model.
        num_layers (int): The number of DR2FWL2Conv layer.
        add_0 (bool): If true, add multiset aggregation involves root nodes.
        add_112 (bool): If true, add all multiset aggregations for triangle 1-1-2.
        add_212 (bool): If true, add all multiset aggregations for triangle 2-1-2.
        add_222 (bool): If true, add all multiset aggregations for triangle 2-2-2.
        add_vv (bool): If true, for each :math::W(u, v), add :math::W(v, v) as additional aggregation.
        eps (bool): Epsilon for distinguishing W(u, v) in aggregation, default is trainable.
        norm_type (str): Normalization type after each layer, choose from ("none", "batch_norm", "layer_norm").
        residual (str): Information aggregation schema for output from each layer. Choose from ("last", "add", "cat").
        drop_prob (float): Dropout probability after each convolutional layer.
    """
    def __init__(self,
                 hidden_channels: int,
                 num_layers: int,
                 add_0: bool = True,
                 add_112: bool = True,
                 add_212: bool = True,
                 add_222: bool = True,
                 add_vv: bool = False,
                 eps: float = 0.,
                 train_eps: bool = False,
                 norm_type: str = "batch_norm",
                 residual: str = "last",
                 drop_prob: float = 0.0):

        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.add_0 = add_0
        self.add_112 = add_112
        self.add_212 = add_212
        self.add_222 = add_222
        self.add_vv = add_vv
        self.initial_eps = eps
        self.train_eps = train_eps
        self.norm_type = norm_type
        self.residual = residual
        self.drop_prob = drop_prob


        gnn = DR2FWL2Conv(self.hidden_channels,
                          self.hidden_channels,
                          self.add_0,
                          self.add_112,
                          self.add_212,
                          self.add_222,
                          self.add_vv,
                          self.initial_eps,
                          self.train_eps,
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





class DR2FWL2KernelZINC(torch.nn.Module):
    r"""Define a 2-Distance Restricted FWL(2) GNN kernel for ZINC dataset by stacking `DR2FWL2ConvZINC`
        layers. This kernel can be further combined with node-level/graph-level
        pooling layers as it is applied on different tasks.
    Args:
        hidden_channels (int): Hidden size of the model.
        num_layers (int): The number of DR2FWL2Conv layer.
        add_0 (bool): If true, add multiset aggregation involves root nodes.
        add_112 (bool): If true, add all multiset aggregations for triangle 1-1-2.
        add_212 (bool): If true, add all multiset aggregations for triangle 2-1-2.
        add_222 (bool): If true, add all multiset aggregations for triangle 2-2-2.
        add_321 (bool): If true, add all multiset aggregations for triangle 3-2-1.
        add_vv (bool): If true, for each :math::W(u, v), add :math::W(v, v) as additional aggregation.
        eps (bool): Epsilon for distinguishing W(u, v) in aggregation, default is trainable.
        norm_type (str): Normalization type after each layer, choose from ("none", "batch_norm", "layer_norm").
        residual (str): Information aggregation schema for output from each layer. Choose from ("last", "add", "cat").
        drop_prob (float): Dropout probability after each convolutional layer.
    """
    def __init__(self,
                 hidden_channels: int,
                 num_layers: int,
                 add_0: bool = True,
                 add_112: bool = True,
                 add_212: bool = True,
                 add_222: bool = True,
                 add_321: bool = True,
                 add_331: bool = True,
                 add_vv: bool = False,
                 eps: float = 0.,
                 train_eps: bool = False,
                 norm_type: str = "batch_norm",
                 residual: str = "last",
                 drop_prob: float = 0.0):

        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.add_0 = add_0
        self.add_112 = add_112
        self.add_212 = add_212
        self.add_222 = add_222
        self.add_321 = add_321
        self.add_331 = add_331
        self.add_vv = add_vv
        self.initial_eps = eps
        self.train_eps = train_eps
        self.norm_type = norm_type
        self.residual = residual
        self.drop_prob = drop_prob


        gnn = DR2FWL2ConvZINC(self.hidden_channels,
                          self.hidden_channels,
                          self.add_0,
                          self.add_112,
                          self.add_212,
                          self.add_222,
                          self.add_321,
                          self.add_331,
                          self.add_vv,
                          self.initial_eps,
                          self.train_eps,
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
                edge_attr3: torch.Tensor,
                edge_index0: torch.Tensor,
                edge_index: torch.LongTensor,
                edge_index2: torch.LongTensor,
                edge_index3: torch.LongTensor,
                triangle_0_1_1: torch.LongTensor,
                triangle_1_1_1: torch.LongTensor,
                triangle_1_1_2: torch.LongTensor,
                triangle_1_2_2: torch.LongTensor,
                triangle_2_2_2: torch.LongTensor,
                triangle_3_2_1: torch.LongTensor,
                triangle_3_3_1: torch.LongTensor,
                inverse_edge_1: torch.LongTensor,
                inverse_edge_2: torch.LongTensor,
                inverse_edge_3: torch.LongTensor,
                edge_emb_list: list = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:


        emb_list0 = [edge_attr0]
        emb_list1 = [edge_attr1]
        emb_list2 = [edge_attr2]
        emb_list3 = [edge_attr3]

        for i in range(self.num_layers):
            edge_attr0 = emb_list0[i]
            edge_attr1 = emb_list1[i]
            edge_attr2 = emb_list2[i]
            edge_attr3 = emb_list3[i]
            if edge_emb_list is not None:
                edge_attr1 += edge_emb_list[i]

            edge_attr0_out, edge_attr1_out, edge_attr2_out, edge_attr3_out = self.gnns[i](edge_attr0,
                                                                                          edge_attr1,
                                                                                          edge_attr2,
                                                                                          edge_attr3,
                                                                                          edge_index0,
                                                                                          edge_index,
                                                                                          edge_index2,
                                                                                          edge_index3,
                                                                                          triangle_0_1_1,
                                                                                          triangle_1_1_1,
                                                                                          triangle_1_1_2,
                                                                                          triangle_1_2_2,
                                                                                          triangle_2_2_2,
                                                                                          triangle_3_2_1,
                                                                                          triangle_3_3_1,
                                                                                          inverse_edge_1,
                                                                                          inverse_edge_2,
                                                                                          inverse_edge_3)

            edge_attr0_out, edge_attr1_out, edge_attr2_out, edge_attr3_out = F.dropout(
                edge_attr0_out, p=self.drop_prob, training=self.training), F.dropout(
                edge_attr1_out, p=self.drop_prob, training=self.training), F.dropout(
                edge_attr2_out, p=self.drop_prob, training=self.training), F.dropout(
                edge_attr3_out, p=self.drop_prob, training=self.training)

            emb_list0.append(edge_attr0_out)
            emb_list1.append(edge_attr1_out)
            emb_list2.append(edge_attr2_out)
            emb_list3.append(edge_attr3_out)

        if self.residual == "last":
            return self.out_lin((emb_list0[-1], emb_list1[-1], emb_list2[-1], emb_list3[-1]))
        elif self.residual == 'add':
            return self.out_lin((sum(emb_list0), sum(emb_list1), sum(emb_list2), sum(emb_list3)))
        elif self.residual == 'cat':
            return self.out_lin((torch.cat(emb_list0, dim=-1),
                                 torch.cat(emb_list1, dim=-1),
                                 torch.cat(emb_list2, dim=-1),
                                 torch.cat(emb_list3, dim=-1)))
