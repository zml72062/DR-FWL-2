import torch.nn.functional as F
from .gnn_conv import *
from .auxiliaries import TwoComponentReLU
from .utils import clones


class DR2FWL2Kernel(torch.nn.Module):
    """
    Define a 2-Distance Restricted FWL(2) GNN kernel by stacking `DR2FWL2Conv`
    layers. This kernel can be further combined with node-level/graph-level
    pooling layers as it is applied on different tasks.
    """
    def __init__(self,
                 in_channels: int,
                 mlp_hidden_channels: int,
                 num_layers: int,
                 dropout: float = 0,
                 residual: Optional[str] = None,
                 norm: Optional[str] = None,
                 relu_first: bool = False,
                 mlp_num_layers: int = 1,
                 mlp_dropout: float = 0.0,
                 eps: float = 0.0,
                 eps2: float = 0.0,
                 train_eps: bool = False,
                 mlp_norm: Optional[str] = None):
        """
        Args:

        in_channels (int): Input edge-feature and 2-hop-edge-feature channels,
        must be the same for 1-hop and 2-hop edges

        mlp_hidden_channels (int): Hidden channels in MLP

        num_layers (int): Number of `DR2FWL2Conv` layers

        dropout (float): Dropout rate after applying a `DR2FWL2Conv` layer and
        ReLU, default 0.0

        residual (Optional[str]): Whether, and how to use skip connection among
        `DR2FWL2Conv` layers, default None, can be None, 'cat' or 'add'

        norm (Optional[str]): Normalization method after applying a `DR2FWL2Conv`
        layer, default None, can be None, 'batch_norm' or 'layer_norm'

        relu_first (bool): Whether to apply ReLU before normalization, default
        False.

        mlp_num_layers (int): Number of non-linear layers in MLP, default 1

        mlp_dropout (float): Dropout rate after applying ReLU in MLP, default 0.0

        eps, eps2 (float): `eps` value for 1-hop / 2-hop edges (see formula),
        default 0.0 both

        train_eps (bool): Whether to treat `eps` and `eps2` as trainable
        parameters, default False

        mlp_norm (Optional[str]): Normalization method for MLP, default None,
        can be None, 'batch_norm' or 'layer_norm'

        """
        super().__init__()
        lin = DR2FWL2Conv(in_channels, mlp_hidden_channels, mlp_num_layers,
                    mlp_dropout, eps, eps2, train_eps, mlp_norm)
        self.lins = clones(lin, num_layers)


        if residual != 'cat':
            self.lins.append(torch.nn.Linear(
                in_channels, in_channels)) # for 1-hop
            self.lins.append(torch.nn.Linear(
                in_channels, in_channels)) # for 2-hop
        else:
            self.lins.append(torch.nn.Linear(
                in_channels *num_layers, in_channels)) # for 1-hop
            self.lins.append(torch.nn.Linear(
                in_channels *num_layers, in_channels)) # for 2-hop

        if norm is None:
            norm = torch.nn.Identity()
        elif norm == 'batch_norm':
            norm = torch.nn.BatchNorm1d(in_channels)
        elif norm == 'layer_norm':
            norm = torch.nn.LayerNorm(in_channels)

        self.norms1 = clones(norm, num_layers)
        self.norms2 = clones(norm, num_layers)


        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.dropout = dropout
        self.relu_first = relu_first
        self.residual = residual

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms1:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()
        for norm in self.norms2:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def forward(self,
                edge_attr: torch.Tensor,
                edge_attr2: torch.Tensor,
                triangle_1_1_1: torch.LongTensor,
                triangle_1_1_2: torch.LongTensor,
                triangle_1_2_2: torch.LongTensor,
                triangle_2_2_2: torch.LongTensor,
                inverse_edge_1: torch.LongTensor,
                inverse_edge_2: torch.LongTensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        if self.residual is not None:
            emb_list = []
            emb_list2 = []

        for i in range(self.num_layers):
            edge_attr, edge_attr2 = self.lins[i](
                edge_attr,
                edge_attr2,
                triangle_1_1_1,
                triangle_1_1_2,
                triangle_1_2_2,
                triangle_2_2_2,
                inverse_edge_1,
                inverse_edge_2
            )
            if self.relu_first:
                edge_attr, edge_attr2 = TwoComponentReLU()(edge_attr, edge_attr2)
            edge_attr, edge_attr2 = (self.norms1[i](edge_attr),
                                     self.norms2[i](edge_attr2))
            if not self.relu_first:
                edge_attr, edge_attr2 = TwoComponentReLU()(edge_attr, edge_attr2)
            edge_attr, edge_attr2 = F.dropout(
                edge_attr, p=self.dropout, training=self.training), F.dropout(
                edge_attr2, p=self.dropout, training=self.training
            )

            if self.residual is not None:
                emb_list.append(edge_attr)
                emb_list2.append(edge_attr2)

        if self.residual is None:
            return self.lins[-2](edge_attr), self.lins[-1](edge_attr2)
        elif self.residual == 'add':
            return self.lins[-2](sum(emb_list)), self.lins[-1](sum(emb_list2))
        elif self.residual == 'cat':
            return (self.lins[-2](torch.cat(emb_list, dim=-1)),
                    self.lins[-1](torch.cat(emb_list2, dim=-1)))
