import torch
import torch.nn as nn
import numpy as np
from data_utils.preprocess import drfwl2_transform, drfwl3_transform
from torch_geometric.seed import seed_everything
import argparse
from pygmmpp.data import Data
from torch.optim import Adam
from models.pool import GraphLevelPooling
from models.gnn_count import DR2FWL2Kernel
import local_fwl2 as lfwl
from tqdm import tqdm
from local_fwl2 import LFWLLayer, SLFWLLayer, SSWLLayer
from networkx import Graph, is_isomorphic
def test_iso(x, y):
    g = Graph()
    g.add_edges_from(x.T)
    h = Graph()
    h.add_edges_from(y.T)
    return is_isomorphic(g, h)

class LFWLWrapper(nn.Module):
    def __init__(self, hidden_channels: int,
                 num_layers: int, model):
        super().__init__()
        self.localfwl2 = lfwl.LocalFWL2(hidden_channels, num_layers, model,
                                        1, None, 'instance')
        self.pooling = lfwl.Pooling(hidden_channels, 1)
    
    def forward(self, batch) -> torch.Tensor:
        return self.pooling(self.localfwl2(
            *lfwl.to_dense(batch.x, batch.edge_index, None, None))).squeeze()
LFWL = lambda x, y: LFWLWrapper(x, y, LFWLLayer)
SLFWL = lambda x, y: LFWLWrapper(x, y, SLFWLLayer)
SSWL = lambda x, y: LFWLWrapper(x, y, SSWLLayer)

class DR3FWL2(nn.Module):
    """
    3-DRFWL(2) GNN
    """
    def __init__(self,
                 hidden_channels: int,
                 num_layers: int,
                 add_0: bool = True,
                 add_112: bool = True,
                 add_212: bool = True,
                 add_222: bool = True,
                 eps: float = 0.,
                 train_eps: bool = False,
                 norm_type: str = "none",
                 norm_between_layers: str = "none",
                 residual: str = "none",
                 drop_prob: float = 0.0):

        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.add_0 = add_0
        self.add_112 = add_112
        self.add_212 = add_212
        self.add_222 = add_222
        self.initial_eps = eps
        self.train_eps = train_eps
        self.norm_type = norm_type
        self.residual = residual
        self.drop_prob = drop_prob

        self.initial_proj = nn.Linear(1, hidden_channels)
        self.distance_encoding = nn.Embedding(3, hidden_channels)

        self.ker = DR2FWL2Kernel(self.hidden_channels,
                                 self.num_layers,
                                 self.initial_eps,
                                 self.train_eps,
                                 self.norm_type,
                                 norm_between_layers,
                                 self.residual,
                                 self.drop_prob)

        self.pool = GraphLevelPooling(hidden_channels)

        self.post_mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels // 2),
                                       nn.ELU(),
                                       nn.Linear(hidden_channels // 2, 1))
        
        self.ker.add_aggr(1, 1, 1)
        if self.add_0:
            self.ker.add_aggr(0, 1, 1)
            self.ker.add_aggr(0, 2, 2)
        if self.add_112:
            self.ker.add_aggr(1, 1, 2)
        if self.add_212:
            self.ker.add_aggr(2, 2, 1)
        if self.add_222:
            self.ker.add_aggr(2, 2, 2)
        self.ker.add_aggr(1, 2, 3)
        self.ker.add_aggr(3, 3, 1)
        self.ker.add_aggr(2, 2, 3)
        self.ker.add_aggr(3, 3, 2)
        self.ker.add_aggr(3, 3, 3)
        self.ker.add_aggr(0, 3, 3)

        self.reset_parameters()

    def reset_parameters(self):
        self.initial_proj.reset_parameters()
        self.distance_encoding.reset_parameters()

        self.ker.reset_parameters()
        for m in self.post_mlp:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def forward(self, batch) -> torch.Tensor:
        edge_indices = [batch.edge_index, batch.edge_index2, batch.edge_index3]
        edge_attrs = [self.initial_proj(batch.x),
                      self.distance_encoding(torch.zeros_like(edge_indices[0][0])),
                      self.distance_encoding(torch.ones_like(edge_indices[1][0])),
                      self.distance_encoding(torch.full_like(edge_indices[2][0], 2))]
        triangles = {
            (1, 1, 1): batch.triangle_1_1_1,
            (1, 1, 2): batch.triangle_1_1_2,
            (2, 2, 1): batch.triangle_2_2_1,
            (2, 2, 2): batch.triangle_2_2_2,
            (1, 2, 3): batch.triangle_1_2_3,
            (3, 3, 1): batch.triangle_3_3_1,
            (2, 2, 3): batch.triangle_2_2_3,
            (3, 3, 2): batch.triangle_3_3_2,
            (3, 3, 3): batch.triangle_3_3_3,
        }
        inverse_edges = [batch.inverse_edge_1, batch.inverse_edge_2, batch.inverse_edge_3]

        edge_attrs = self.ker(edge_attrs,
                              edge_indices,
                              triangles,
                              inverse_edges)


        x = self.pool(edge_attrs, edge_indices, batch.num_nodes, torch.zeros((1,), dtype=torch.long))
        x = self.post_mlp(x).squeeze()
        return x


class DR2FWL2(nn.Module):
    def __init__(self,
                 hidden_channels: int,
                 num_layers: int,
                 add_0: bool = True,
                 add_112: bool = True,
                 add_212: bool = True,
                 add_222: bool = True,
                 eps: float = 0.,
                 train_eps: bool = False,
                 norm_type: str = "none",
                 norm_between_layers: str = "none",
                 residual: str = "none",
                 drop_prob: float = 0.0):

        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.add_0 = add_0
        self.add_112 = add_112
        self.add_212 = add_212
        self.add_222 = add_222
        self.initial_eps = eps
        self.train_eps = train_eps
        self.norm_type = norm_type
        self.residual = residual
        self.drop_prob = drop_prob

        self.initial_proj = nn.Linear(1, hidden_channels)
        self.distance_encoding = nn.Embedding(2, hidden_channels)

        self.ker = DR2FWL2Kernel(self.hidden_channels,
                                 self.num_layers,
                                 self.initial_eps,
                                 self.train_eps,
                                 self.norm_type,
                                 norm_between_layers,
                                 self.residual,
                                 self.drop_prob)

        self.pool = GraphLevelPooling(hidden_channels)

        self.post_mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels // 2),
                                       nn.ELU(),
                                       nn.Linear(hidden_channels // 2, 1))
        
        self.ker.add_aggr(1, 1, 1)
        if self.add_0:
            self.ker.add_aggr(0, 1, 1)
            self.ker.add_aggr(0, 2, 2)
        if self.add_112:
            self.ker.add_aggr(1, 1, 2)
        if self.add_212:
            self.ker.add_aggr(2, 2, 1)
        if self.add_222:
            self.ker.add_aggr(2, 2, 2)

        self.reset_parameters()

    def reset_parameters(self):
        self.initial_proj.reset_parameters()
        self.distance_encoding.reset_parameters()

        self.ker.reset_parameters()
        for m in self.post_mlp:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def forward(self, batch) -> torch.Tensor:
        edge_indices = [batch.edge_index, batch.edge_index2]
        edge_attrs = [self.initial_proj(batch.x),
                      self.distance_encoding(torch.zeros_like(edge_indices[0][0])),
                      self.distance_encoding(torch.ones_like(edge_indices[1][0]))]
        triangles = {
            (1, 1, 1): batch.triangle_1_1_1,
            (1, 1, 2): batch.triangle_1_1_2,
            (2, 2, 1): batch.triangle_2_2_1,
            (2, 2, 2): batch.triangle_2_2_2,
        }
        inverse_edges = [batch.inverse_edge_1, batch.inverse_edge_2]

        edge_attrs = self.ker(edge_attrs,
                              edge_indices,
                              triangles,
                              inverse_edges)


        x = self.pool(edge_attrs, edge_indices, batch.num_nodes, torch.zeros((1,), dtype=torch.long))
        x = self.post_mlp(x).squeeze()
        return x
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='random seed.')
parser.add_argument('--graph', type=int, default=0)
parser.add_argument('--epochs', type=int, default=0)
args = parser.parse_args()

base_graph = [[[0, 1, 1, 2, 2, 3, 3, 0, 1, 3, 0, 2],
            [1, 0, 2, 1, 3, 2, 0, 3, 3, 1, 2, 0]] # 4-clique, give rise to shrikhande vs 4x4-rook
, [[0,1,1,2,2,0,1,3,2,3,3,4,4,5,3,5,4,6,5,6],
              [1,0,2,1,0,2,3,1,3,2,4,3,5,4,5,3,6,4,6,5]] # figure 4 of SWL paper
, [[0,1,1,2,2,0,0,3,0,4,3,4,3,7,4,7,5,6,6,7,5,7],
              [1,0,2,1,0,2,3,0,4,0,4,3,7,3,7,4,6,5,7,6,7,5]] # figure 5 of SWL paper
, [[0,1,1,2,2,0,1,3,2,3,3,4,4,5,5,3,5,6,6,7,7,5,6,8,7,8],
              [1,0,2,1,0,2,3,1,3,2,4,3,5,4,3,5,6,5,7,6,5,7,8,6,8,7]] # figure 6 of SWL paper
, [[0,1,0,2,1,3,2,3,2,4,4,5,5,3,4,6,6,7,7,5],
              [1,0,2,0,3,1,3,2,4,2,5,4,3,5,6,4,7,6,5,7]] # figure 7 of SWL paper
, [[0,1,1,2,2,0,1,3,3,4,4,1,2,4,4,5,5,2],
              [1,0,2,1,0,2,3,1,4,3,1,4,4,2,5,4,2,5]] # figure 8 of SWL paper, SSWL fail
, [[0,1,1,3,3,5,5,4,4,2,2,0,0,6,6,4,1,7,7,5],
   [1,0,3,1,5,3,4,5,2,4,0,2,6,0,4,6,7,1,5,7]] # figure 9
]
for model_name in ['LFWL', 'SLFWL', 'SSWL', 'DR2FWL2', 'DR3FWL2']:
    seed_everything(args.seed)
    model = eval(model_name)(
        100, 5
    )

    def build_graph(edge_index: np.ndarray, 
                    transform=None):
        g, gx, h, hx = get_furer_graph_pair(edge_index)
        g, h = Data(x=torch.from_numpy(gx), 
             edge_index=torch.from_numpy(g)), \
                Data(x=torch.from_numpy(hx), 
             edge_index=torch.from_numpy(h))
        if transform is not None:
            g, h = transform(g), transform(h)
        return g, h

    from software.furer_graph.furer import get_furer_graph_pair
    if model_name == 'DR3FWL2':
        transform = drfwl3_transform()
    elif model_name == 'DR2FWL2':
        transform = drfwl2_transform()
    else:
        transform = None
    g, h = build_graph(np.array(
            base_graph[args.graph], dtype=np.int64
        ), transform)
    optimizer = Adam(model.parameters(), lr=0.001)
    print('Is isomorphic: ', test_iso(g.edge_index.numpy(), h.edge_index.numpy()))
    for i in tqdm(range(args.epochs)):
        optimizer.zero_grad()
        l = nn.BCEWithLogitsLoss()(torch.cat([model(g).reshape(-1), model(h).reshape(-1)]),
                            torch.tensor([0,1], dtype=torch.float32))
        l.backward()
        optimizer.step()
    print(model_name)
    print('G: ', model(g).item())
    print('H: ', model(h).item())
    print('Can' if model(g).item() != model(h).item() else 'Cannot')

