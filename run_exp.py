import os.path as osp
import shutil
import numpy as np
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_scatter import scatter_mean, scatter_max
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv, GINConv, global_add_pool
import torch_geometric.transforms as T
from models.gnn_count import DR2FWL2Kernel
from pygmmpp.datasets import EXP
from data_utils.batch import collate
from pygmmpp.data import DataLoader
from data_utils.preprocess import drfwl2_transform
from models.pool import GraphLevelPooling
from pygmmpp.utils import compose
from pygmmpp.nn.gin_conv import GINConv
from pygmmpp.nn.pool import GlobalPool


class GINModel(nn.Module):
    def __init__(self, in_channels: int,
                 num_layers: int):
        super().__init__()
        self.node_transform = nn.Linear(2, in_channels)

        self.convs = nn.ModuleList() 
        for _ in range(num_layers):
            self.convs.append(GINConv(nn.Sequential(
            nn.Linear(in_channels, 2*in_channels),
            nn.BatchNorm1d(2*in_channels),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2*in_channels, in_channels)
        )))
            
        self.pool = GlobalPool()
        self.post_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(in_channels // 2, 2)
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        self.node_transform.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for m in self.post_mlp:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    
    def forward(self, batch) -> torch.Tensor:
        x, edge_index, batchv = (self.node_transform(batch.x),
                                    batch.edge_index,
                                    batch.batch0)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(F.relu(x), 0.5, training=self.training)
        
        x = self.pool(x, batchv)
        return F.log_softmax(self.post_mlp(x), dim=1)


class EXPModel(nn.Module):
    def __init__(self,
                 hidden_channels: int,
                 num_layers: int,
                 add_0: bool = True,
                 add_112: bool = True,
                 add_212: bool = True,
                 add_222: bool = True,
                 eps: float = 0.,
                 train_eps: bool = False,
                 norm_type: str = "batch_norm",
                 norm_between_layers: str = "batch_norm",
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

        self.node_transform = nn.Linear(2, self.hidden_channels)

        self.ker = DR2FWL2Kernel(self.hidden_channels,
                                 self.num_layers,
                                 self.initial_eps,
                                 self.train_eps,
                                 self.norm_type,
                                 norm_between_layers,
                                 self.residual,
                                 self.drop_prob)

        self.pool = GraphLevelPooling()

        self.post_mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels // 2),
                                       nn.ELU(),
                                       nn.Linear(hidden_channels // 2, 2))
        
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
        self.node_transform.reset_parameters()
        self.ker.reset_parameters()
        for m in self.post_mlp:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def forward(self, batch) -> torch.Tensor:
        edge_indices = [batch.edge_index, batch.edge_index2]
        edge_attrs = [self.node_transform(batch.x),
                      self.node_transform(batch.x[batch.edge_index[0]]) +
                      self.node_transform(batch.x[batch.edge_index[1]]),
                      self.node_transform(batch.x[batch.edge_index2[0]]) +
                      self.node_transform(batch.x[batch.edge_index2[1]])
                      ]
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


        x = self.pool(*edge_attrs, *edge_indices, batch.num_nodes, batch.batch0)
        x = self.post_mlp(x)
        x = F.log_softmax(x, dim=1)
        return x


parser = argparse.ArgumentParser(description='DRFWL(2) for EXP/CEXP datasets')
parser.add_argument('--layers', type=int, default=5)   # Number of GNN layers
parser.add_argument('--width', type=int, default=64)    # Dimensionality of GNN embeddings
parser.add_argument('--epochs', type=int, default=10)    # Number of training epochs
parser.add_argument('--dataset', type=str, default='exp')    # Dataset being used
parser.add_argument('--learnRate', type=float, default=0.001)   # Learning Rate
args = parser.parse_args()


def print_or_log(input_data, log=False, log_file_path="Debug.txt"):
    if not log:  # If not logging, we should just print
        print(input_data)
    else:  # Logging
        log_file = open(log_file_path, "a+")
        log_file.write(str(input_data) + "\r\n")
        log_file.close()  # Keep the file available throughout execution


class MyPreTransform(object):
    def __call__(self, data):
        data.x = F.one_hot(data.x[:, 0], num_classes=2).to(torch.float)  # Convert node labels to one-hot
        return data


# Command Line Arguments
DATASET = args.dataset
LAYERS = args.layers
EPOCHS = args.epochs
WIDTH = args.width
LEARNING_RATE = args.learnRate

MODEL = f"DRFWL(2)-GNN-"


if LEARNING_RATE != 0.001:
    MODEL = MODEL+"lr"+str(LEARNING_RATE)+"-"

BATCH = 20
MODULO = 4
MOD_THRESH = 1

path = 'datasets/' + DATASET
 
dataset = EXP(root=path, pre_transform=compose([MyPreTransform(), drfwl2_transform()]))


device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

model = EXPModel(WIDTH,
                 LAYERS,
                 add_112=True,
                 add_212=True,
                 add_222=False,
                 norm_type='none',
                 norm_between_layers='none',
                 residual='last').to(device)
# model = GINModel(WIDTH, LAYERS).to(device)

def train(epoch, loader, optimizer):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)


def val(loader):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        loss_all += F.nll_loss(model(data), data.y, reduction='sum').item()
    return loss_all / len(loader.dataset)


def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        nb_trials = 1   # Support majority vote, but single trial is default
        successful_trials = torch.zeros_like(data.y)
        for i in range(nb_trials):  # Majority Vote
            pred = model(data).max(1)[1]
            successful_trials += pred.eq(data.y)
        successful_trials = successful_trials > (nb_trials // 2)
        correct += successful_trials.sum().item()
    return correct / len(loader.dataset)

acc = []
tr_acc = []
#SPLITS = 2
SPLITS = 10
tr_accuracies = np.zeros((EPOCHS, SPLITS))
tst_accuracies = np.zeros((EPOCHS, SPLITS))
tst_exp_accuracies = np.zeros((EPOCHS, SPLITS))
tst_lrn_accuracies = np.zeros((EPOCHS, SPLITS))

for i in range(SPLITS):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, min_lr=LEARNING_RATE)

    n = len(dataset) // SPLITS
    test_mask = torch.zeros(len(dataset), dtype=torch.bool)
    test_exp_mask = torch.zeros(len(dataset), dtype=torch.bool)
    test_lrn_mask = torch.zeros(len(dataset), dtype=torch.bool)

    test_mask[i * n:(i + 1) * n] = 1 # Now set the masks
    learning_indices = [x for idx, x in enumerate(range(n * i, n * (i+1))) if x % MODULO <= MOD_THRESH]
    test_lrn_mask[learning_indices] = 1
    exp_indices = [x for idx, x in enumerate(range(n * i, n * (i+1))) if x % MODULO > MOD_THRESH]
    test_exp_mask[exp_indices] = 1

    # Now load the datasets
    test_dataset = dataset[test_mask]
    test_exp_dataset = dataset[test_exp_mask]
    test_lrn_dataset = dataset[test_lrn_mask]
    train_dataset = dataset[~test_mask]

    n = len(train_dataset) // SPLITS
    val_mask = torch.zeros(len(train_dataset), dtype=torch.bool)
    val_mask[i * n:(i + 1) * n] = 1
    val_dataset = train_dataset[val_mask]
    train_dataset = train_dataset[~val_mask]

    val_loader = DataLoader(val_dataset, collator=collate, batch_size=BATCH)
    test_loader = DataLoader(test_dataset, collator=collate, batch_size=BATCH)
    test_exp_loader = DataLoader(test_exp_dataset, collator=collate, batch_size=BATCH) # These are the new test splits
    test_lrn_loader = DataLoader(test_lrn_dataset, collator=collate, batch_size=BATCH)
    train_loader = DataLoader(train_dataset, collator=collate, batch_size=BATCH, shuffle=True)


    print_or_log('---------------- Split {} ----------------'.format(i),
                 log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
    best_val_loss, test_acc = 100, 0
    for epoch in tqdm(range(EPOCHS)):
        lr = scheduler.optimizer.param_groups[0]['lr']
        train_loss = train(epoch, train_loader, optimizer)
        val_loss = val(val_loader)
        scheduler.step(val_loss)
        if best_val_loss >= val_loss:
            best_val_loss = val_loss
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        test_exp_acc = test(test_exp_loader)
        test_lrn_acc = test(test_lrn_loader)
        tr_accuracies[epoch, i] = train_acc
        tst_accuracies[epoch, i] = test_acc
        tst_exp_accuracies[epoch, i] = test_exp_acc
        tst_lrn_accuracies[epoch, i] = test_lrn_acc
        print_or_log('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
              'Val Loss: {:.7f}, Test Acc: {:.7f}, Exp Acc: {:.7f}, Lrn Acc: {:.7f}, Train Acc: {:.7f}'.format(
                  epoch+1, lr, train_loss, val_loss, test_acc, test_exp_acc, test_lrn_acc, train_acc),log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
    acc.append(test_acc)
    tr_acc.append(train_acc)

acc = torch.tensor(acc)
tr_acc = torch.tensor(tr_acc)
print_or_log('---------------- Final Result ----------------',
             log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
print_or_log('Mean: {:7f}, Std: {:7f}'.format(acc.mean(), acc.std()),
             log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
print_or_log('Tr Mean: {:7f}, Std: {:7f}'.format(tr_acc.mean(), tr_acc.std()),
             log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
