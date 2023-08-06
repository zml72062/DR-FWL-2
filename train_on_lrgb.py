import sklearn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from software.lrgb_graph.evaluator import Evaluator
from software.lrgb_graph import PeptidesFunctionalDataset, PeptidesStructuralDataset
import argparse
from pygmmpp.utils import compose
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from pygmmpp.data import DataLoader
import train
from models.gnn_count import DR2FWL2Kernel
from models.pool import GraphLevelPooling
from data_utils.batch import collate
from data_utils.preprocess import drfwl2_transform, drfwl3_transform
import train_utils
from pygmmpp.nn.gin_conv import GINEConv
from pygmmpp.nn.pool import GlobalPool
from tqdm import tqdm

class GINEModel(nn.Module):
    def __init__(self, in_channels: int,
                 num_layers: int,
                 num_tasks: int):
        super().__init__()
        self.atom_encoder = AtomEncoder(in_channels)
        self.bond_encoder = BondEncoder(in_channels)
        self.num_tasks = num_tasks

        self.convs = nn.ModuleList() 
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GINEConv(nn.Sequential(
            nn.Linear(in_channels, 2*in_channels),
            nn.BatchNorm1d(2*in_channels),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2*in_channels, in_channels)
        )))
            self.norms.append(nn.BatchNorm1d(in_channels))
            
        self.pool = GlobalPool()
        self.post_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(in_channels // 2, num_tasks)
        )
    
    def forward(self, batch) -> torch.Tensor:
        x, edge_index, edge_attr, batchv = (self.atom_encoder(batch.x),
                                    batch.edge_index,
                                    self.bond_encoder(batch.edge_attr),
                                    batch.batch0)
        for (norm, conv) in zip(self.norms, self.convs):
            x = norm(conv(x, edge_index, edge_attr=edge_attr))
            x = F.dropout(F.relu(x), 0.5, training=self.training)
        
        x = self.pool(x, batchv)
        return self.post_mlp(x).reshape((-1, self.num_tasks))

import local_fwl2 as lfwl
from local_fwl2 import LFWLLayer, SLFWLLayer, SSWLPlusLayer, SSWLLayer

class LFWLWrapper(nn.Module):
    def __init__(self, hidden_channels: int,
                 num_layers: int,
                 num_tasks: int, model):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_channels)
        self.bond_encoder = BondEncoder(hidden_channels)
        self.localfwl2 = lfwl.LocalFWL2(hidden_channels, num_layers, model,
                                        hidden_channels, hidden_channels, 'instance')
        self.pooling = lfwl.Pooling(hidden_channels, num_tasks)
    
    def forward(self, batch) -> torch.Tensor:
        return self.pooling(self.localfwl2(
            *lfwl.to_dense(F.relu(self.atom_encoder(batch.x)), 
                           batch.edge_index, 
                           F.relu(self.bond_encoder(batch.edge_attr)), 
                           batch.batch0)))

class OGBMOLModel3(nn.Module):
    def __init__(self,
                 hidden_channels: int,
                 num_layers: int,
                 num_tasks: int,
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
        self.num_tasks = num_tasks
        self.add_0 = add_0
        self.add_112 = add_112
        self.add_212 = add_212
        self.add_222 = add_222
        self.initial_eps = eps
        self.train_eps = train_eps
        self.norm_type = norm_type
        self.residual = residual
        self.drop_prob = drop_prob

        self.atom_encoder = AtomEncoder(hidden_channels)
        self.bond_encoder = BondEncoder(hidden_channels)

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
                                       nn.Linear(hidden_channels // 2, num_tasks))
        
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
        self.ker.reset_parameters()
        for m in self.post_mlp:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def forward(self, batch) -> torch.Tensor:
        edge_indices = [batch.edge_index, batch.edge_index2, batch.edge_index3]
        edge_attrs = [self.atom_encoder(batch.x),
                      self.bond_encoder(batch.edge_attr),
                      self.atom_encoder(batch.x[batch.edge_index2[0]]) +
                      self.atom_encoder(batch.x[batch.edge_index2[1]]),
                      self.atom_encoder(batch.x[batch.edge_index3[0]]) +
                      self.atom_encoder(batch.x[batch.edge_index3[1]])
                      ]
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


        x = self.pool(edge_attrs, edge_indices, batch.num_nodes, batch.batch0)
        x = self.post_mlp(x).reshape(-1, self.num_tasks)
        return x

class OGBMOLModel(nn.Module):
    def __init__(self,
                 hidden_channels: int,
                 num_layers: int,
                 num_tasks: int,
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
        self.num_tasks = num_tasks
        self.add_0 = add_0
        self.add_112 = add_112
        self.add_212 = add_212
        self.add_222 = add_222
        self.initial_eps = eps
        self.train_eps = train_eps
        self.norm_type = norm_type
        self.residual = residual
        self.drop_prob = drop_prob

        self.atom_encoder = AtomEncoder(hidden_channels)
        self.bond_encoder = BondEncoder(hidden_channels)

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
                                       nn.Linear(hidden_channels // 2, num_tasks))
        
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
        self.ker.reset_parameters()
        for m in self.post_mlp:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def forward(self, batch) -> torch.Tensor:
        edge_indices = [batch.edge_index, batch.edge_index2]
        edge_attrs = [self.atom_encoder(batch.x),
                      self.bond_encoder(batch.edge_attr),
                      self.atom_encoder(batch.x[batch.edge_index2[0]]) +
                      self.atom_encoder(batch.x[batch.edge_index2[1]])
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


        x = self.pool(edge_attrs, edge_indices, batch.num_nodes, batch.batch0)
        x = self.post_mlp(x).reshape(-1, self.num_tasks)
        return x


def epoch(model, 
          num_classes,
          loader,
          dataset_len, 
          device, 
          evaluator,
          metric_name,
          optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()
    
    loss = 0.0
    y_true, y_pred = [], []
    for batch in tqdm(loader):
        if optimizer is not None:
            optimizer.zero_grad()
        batch = batch.to(device)
        y = batch.y.reshape(-1, num_classes).to(torch.float)
        y_true.append(y.detach().cpu())
        is_labeled = y == y
        pred = model(batch)
        y_pred.append(pred.detach().cpu())
        batch_loss = (nn.BCEWithLogitsLoss if metric_name == 'ap'
                      else nn.L1Loss)()(pred[is_labeled], 
                                          y[is_labeled])
        if optimizer is not None:
            batch_loss.backward()
            optimizer.step()
        with torch.no_grad():
            loss += batch_loss.item() * y.shape[0]
        torch.cuda.empty_cache()
    return loss/dataset_len, evaluator.eval({"y_true": torch.cat(y_true, dim=0),
                              "y_pred": torch.cat(y_pred, dim=0)})[metric_name]

import sys

def run(epochs, 
        num_classes,
        model, 
        train_loader, 
        valid_loader,
        test_loader,
        train_set,
        valid_set,
        test_set,
        evaluator,
        metric_name,
        device, 
        optimizer, 
        lr_scheduler=None, 
        choose_best='max',
        log_file=sys.stdout):
    
    model.to(device)

    best_val_metric = 1e6 if choose_best == 'min' else 0
    best_test_metric = 0

    for idx in range(epochs):
        print(idx, file=sys.stderr)
        train_loss, train_metric = epoch(model, num_classes, train_loader, 
                                         len(train_set), device, evaluator, metric_name, optimizer)
        val_loss, val_metric = epoch(model, num_classes, valid_loader, len(valid_set),
                                     device, evaluator, metric_name, None)
        test_loss, test_metric = epoch(model, num_classes, test_loader, len(test_set),
                                       device, evaluator, metric_name, None)
        if choose_best == 'max' and val_metric > best_val_metric:
            best_val_metric = val_metric
            best_test_metric = test_metric
        elif choose_best == 'min' and val_metric < best_val_metric:
            best_val_metric = val_metric
            best_test_metric = test_metric
        if lr_scheduler is not None:
            if lr_scheduler.__class__ != ReduceLROnPlateau:
                lr_scheduler.step()
            else:
                lr_scheduler.step(val_metric)

        if log_file is not None:
            print("Epoch %d: " % idx, file=log_file)
            print("Training Loss: %f    Training %s: %f" % (train_loss, metric_name.upper(), train_metric), file=log_file)
            print("Validation Loss: %f    Validation %s: %f" % (val_loss, metric_name.upper(), val_metric), file=log_file)
            print("Test Loss: %f    Test %s: %f" % (test_loss, metric_name.upper(), test_metric), file=log_file)

    return best_test_metric

# General settings.
parser = argparse.ArgumentParser(description='DRFWL2GNNs for OGBMOL graphs')

"""
Definition for command-line arguments.
"""
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--config-path', type=str, default='configs/lrgb.json',
                    help='Path of the configure file.')
parser.add_argument('--save-dir', type=str, default='results/ogbmol',
                    help='Directory to save the result.')
parser.add_argument('--copy-data', action='store_true',
                    help='Whether to copy raw data to result directory.')
parser.add_argument('--name', type=str, default='Functional',
                    help='Functional/Structural')

args = parser.parse_args()

def train_on_ogb(seed):
    """
    Load configure file.
    """
    additional_args = train_utils.load_json(args.config_path)
    loader = train_utils.json_loader(additional_args)
    
    """
    Copy necessary info for reproducing result.
    """
    if args.copy_data:
        dir = train_utils.copy(args.config_path, args.save_dir, True, loader.dataset.root)
        root = dir
    else:
        dir = train_utils.copy(args.config_path, args.save_dir)
        root = loader.dataset.root
    
    """
    Set random seed.
    """
    train.seed_everything(seed)


    """
    Get and process the dataset.
    """
    metric_names = {
        'Functional': 'ap',
        'Structural': 'mae'
    }
    num_tasks = {
        'Functional': 10,
        'Structural': 11
    }
    dataset = eval(f'Peptides{args.name}Dataset')(root=loader.dataset.root,
                                                  pre_transform=drfwl2_transform()
                                                  )
    split = dataset.get_idx_split()

    train_dataset = dataset[split['train']]
    val_dataset = dataset[split['val']]
    test_dataset = dataset[split['test']]

    """
    Load the dataset.
    """
    train_loader = DataLoader(train_dataset, batch_size=loader.train.batch_size,
                              shuffle=True, collator=collate)
    val_loader = DataLoader(val_dataset, batch_size=loader.train.batch_size,
                            shuffle=False, collator=collate)
    test_loader = DataLoader(test_dataset, batch_size=loader.train.batch_size,
                             shuffle=False, collator=collate)

    """
    Set the device.
    """
    device = f"cuda:{loader.train.cuda}" if loader.train.cuda != -1 else "cpu"

    """
    Get the model.
    """
    model = OGBMOLModel(
                        loader.model.hidden_channels,
                        loader.model.num_layers,
                        num_tasks[args.name],
                        loader.model.add_0,
                        loader.model.add_112,
                        loader.model.add_212,
                        loader.model.add_222,
                        loader.model.eps,
                        loader.model.train_eps,
                        loader.model.norm,
                        loader.model.in_layer_norm,
                        loader.model.residual,
                        loader.model.dropout)
    # model = GINEModel(loader.model.hidden_channels,
    #                   loader.model.num_layers,
    #                   dataset.num_tasks)

    # model = EGNN(loader.model.hidden_channels, dataset.num_tasks,
    #              loader.model.num_layers, loader.model.dropout, 'gin', True)

    # print("# of params: ", sum([f.numel() for f in model.parameters()]))
    """
    Get the optimizer.
    """
    optimizer = Adam(model.parameters(), lr=loader.train.lr,
                     betas=(loader.train.adam_beta1, loader.train.adam_beta2),
                     eps=loader.train.adam_eps,
                     weight_decay=loader.train.l2_penalty)
    """
    Get the LR scheduler.
    """
    scheduler = ReduceLROnPlateau(optimizer, 'max' if args.name == 'Functional' else 'min',
                                  factor=loader.train.lr_reduce_factor,
                                  patience=loader.train.lr_reduce_patience,
                                  min_lr=loader.train.lr_reduce_min)
    
    """
    Run the training script.
    """
    return run(loader.train.epochs,
               num_tasks[args.name],
              model,
              train_loader,
              val_loader,
              test_loader,
              train_dataset,
              val_dataset,
              test_dataset,
              Evaluator(metric_names[args.name]),
              metric_names[args.name],
              device,
              optimizer,
              scheduler,
              'max' if args.name == 'Functional' else 'min')

if __name__ == '__main__':
    print(f"Use file {args.config_path}")
    print(f"Train on Peptides{args.name} dataset")
    print(f"Use seed {args.seed}")
    print(train_on_ogb(args.seed))
