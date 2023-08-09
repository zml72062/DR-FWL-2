import argparse
from pygmmpp.utils import compose
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pygmmpp.datasets import qm9
from pygmmpp.data import Data, DataLoader
import train
from models.gnn_count import DR2FWL2Kernel
from models.pool import GraphLevelPooling
from data_utils.batch import collate
from data_utils.preprocess import drfwl2_transform
import train_utils
import time
import local_fwl2 as lfwl
from local_fwl2 import LFWLLayer, SLFWLLayer, SSWLPlusLayer, SSWLLayer

class LFWLWrapper(nn.Module):
    def __init__(self, hidden_channels: int,
                 num_layers: int, model):
        super().__init__()
        self.localfwl2 = lfwl.LocalFWL2(hidden_channels, num_layers, model,
                                        11, 5, 'instance')
        self.pooling = lfwl.Pooling(hidden_channels, 1)
    
    def forward(self, batch) -> torch.Tensor:
        return self.pooling(self.localfwl2(
            *lfwl.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch0))).squeeze()
    

class QM9Model(nn.Module):
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

        self.initial_proj0 = nn.Linear(11, hidden_channels) # for 0-hop
        self.initial_proj1 = nn.Linear(16, hidden_channels) # for 1-hop
        self.initial_proj2 = nn.Linear(11, hidden_channels) # for 2-hop

        self.ker = DR2FWL2Kernel(self.hidden_channels,
                                 self.num_layers,
                                 self.initial_eps,
                                 self.train_eps,
                                 self.norm_type,
                                 norm_between_layers,
                                 self.residual,
                                 self.drop_prob)

        self.pool = GraphLevelPooling(self.hidden_channels)

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
        self.initial_proj0.reset_parameters()
        self.initial_proj1.reset_parameters()
        self.initial_proj2.reset_parameters()

        self.ker.reset_parameters()
        for m in self.post_mlp:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def forward(self, batch) -> torch.Tensor:
        edge_indices = [batch.edge_index, batch.edge_index2]
        edge_attrs = [self.initial_proj0(batch.x),
                      self.initial_proj1(
                        torch.cat(
                            [batch.edge_attr, batch.x[batch.edge_index[0]]
                             + batch.x[batch.edge_index[1]]], dim=1
                        )
                      ),
                      self.initial_proj2(
                            batch.x[batch.edge_index2[0]] + 
                            batch.x[batch.edge_index2[1]]
                      )]
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
        x = self.post_mlp(x).squeeze()
        return x

class QM9Transform:
    """
    Select a target to train against, and do (optional) unit conversion,
    for QM9 dataset.
    """
    def __init__(self, target: int,
                 pre_convert: bool = False):
        self.target = target
        self.pre_convert = pre_convert

    def __call__(self, data: Data):
        data.y = data.y[:, self.target]  # Specify target: 0 = mu for example
        if self.pre_convert:  # convert back to original units
            data.y = data.y / qm9.conversion[self.target]
        return data

class Distance:
    r"""Saves the Euclidean distance of linked nodes in its edge attributes.

    Args:
        norm (bool, optional): If set to :obj:`False`, the output will not be
            normalized to the interval :math:`[0, 1]`. (default: :obj:`True`)
        max_value (float, optional): If set and :obj:`norm=True`, normalization
            will be performed based on this value instead of the maximum value
            found in the data. (default: :obj:`None`)
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
    """
    def __init__(self, norm=True, max_value=None, cat=True, relative_pos=False, 
                 squared=False):
        self.norm = norm
        self.max = max_value
        self.cat = cat
        self.relative_pos = relative_pos
        self.squared = squared

    def __call__(self, data):
        if type(data) == dict:
            return {key: self.__call__(data_) for key, data_ in data.items()}

        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        if self.squared:
            dist = ((pos[col] - pos[row]) ** 2).sum(1).view(-1, 1)
        else:
            dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)

        if self.norm and dist.numel() > 0:
            dist = dist / (dist.max() if self.max is None else self.max)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = dist

        if self.relative_pos:
            relative_pos = pos[col] - pos[row]
            data.edge_attr = torch.cat([data.edge_attr, relative_pos], dim=-1)

        return data

    def __repr__(self):
        return '{}(norm={}, max_value={})'.format(self.__class__.__name__,
                                                  self.norm, self.max)


# General settings.
parser = argparse.ArgumentParser(description='DRFWL2GNNs for QM9 graphs')

"""
Definition for command-line arguments.
"""
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--config-path', type=str, default='configs/qm9.json',
                    help='Path of the configure file.')
parser.add_argument('--save-dir', type=str, default='results/qm9',
                    help='Directory to save the result.')
parser.add_argument('--copy-data', action='store_true',
                    help='Whether to copy raw data to result directory.')
parser.add_argument('--lfwl', type=str, default='none',
                    help='Which local FWL(2) variant to use, can be '
                    'SSWL/SSWLPlus/LFWL/SLFWL/none')
parser.add_argument('--cuda', type=int, default=0)

args = parser.parse_args()

def train_on_qm9(seed):
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
    before_preprocessing = time.time()
    dataset = qm9.QM9(
        root, 
        transform=compose(
            [
                QM9Transform(loader.dataset.target, loader.preprocess.convert=='pre'), 
                Distance(norm=loader.preprocess.not_normalize_dist==False, 
                        relative_pos=loader.preprocess.use_relative_pos, 
                        squared=loader.preprocess.squared_dist)
            ]
        ), 
        pre_transform=drfwl2_transform() if args.lfwl == "none" else None
    )
    after_preprocessing = time.time()
    print("Pre-processing time, ", after_preprocessing - before_preprocessing)
    ### Must shuffle first, and normalize next, otherwise leads to
    ### data leaking.
    dataset = dataset.shuffle()
    
    # Normalize targets to mean = 0 and std = 1. data leaking?
    tenpercent = int(len(dataset) * 0.1)

    ### Select validation and training split
    mean = dataset.data_batch.y[dataset.indices[tenpercent:]].mean(dim=0)
    std = dataset.data_batch.y[dataset.indices[tenpercent:]].std(dim=0)

    print(f"Mean: {mean[loader.dataset.target]}, Std: {std[loader.dataset.target]}")

    dataset.data_batch.y = (dataset.data_batch.y - mean) / std

    test_dataset = dataset[:tenpercent]
    val_dataset = dataset[tenpercent:2 * tenpercent]
    train_dataset = dataset[2 * tenpercent:]

    """
    Load the dataset.
    """
    train_loader = DataLoader(train_dataset, batch_size=loader.train.batch_size,
                              shuffle=True, collator=collate) if args.lfwl == 'none'\
                else DataLoader(train_dataset, batch_size=loader.train.batch_size,
                              shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=loader.train.batch_size,
                            shuffle=False, collator=collate) if args.lfwl == 'none'\
                else DataLoader(val_dataset, batch_size=loader.train.batch_size,
                            shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=loader.train.batch_size,
                             shuffle=False, collator=collate) if args.lfwl == 'none'\
                else DataLoader(test_dataset, batch_size=loader.train.batch_size,
                             shuffle=False)

    """
    Set the device.
    """
    device = f"cuda:{args.cuda}" if args.cuda != -1 else "cpu"

    """
    Get the model.
    """
    model = QM9Model(
                        loader.model.hidden_channels,
                        loader.model.num_layers,
                        loader.model.add_0,
                        loader.model.add_112,
                        loader.model.add_212,
                        loader.model.add_222,
                        loader.model.eps,
                        loader.model.train_eps,
                        loader.model.norm,
                        loader.model.in_layer_norm,
                        loader.model.residual,
                        loader.model.dropout) if args.lfwl == 'none' else \
       LFWLWrapper(loader.model.hidden_channels, 
                            loader.model.num_layers,
                            eval(f"{args.lfwl}Layer"))


    print("# of params: ", sum([f.numel() for f in model.parameters()]))

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
    scheduler = ReduceLROnPlateau(optimizer, 'min',
                                  factor=loader.train.lr_reduce_factor,
                                  patience=loader.train.lr_reduce_patience,
                                  min_lr=loader.train.lr_reduce_min)
    """
    Get the loss and metric.
    """
    pred_fn = lambda model, batch: model(batch)
    truth_fn = lambda batch: batch.y
    loss_fn = nn.L1Loss()
    metric = lambda pred, truth: (pred - truth).abs().mean(dim=0)

    """
    Run the training script.
    """
    return train.run(loader.train.epochs,
              model,
              train_loader,
              val_loader,
              test_loader,
              train_dataset,
              val_dataset,
              test_dataset,
              pred_fn,
              truth_fn,
              loss_fn,
              metric,
              'MAE',
              lambda batch: batch.num_graphs,
              device,
              optimizer,
              scheduler,
              'min')

if __name__ == '__main__':
    print(f"Use file {args.config_path}")
    print(train_on_qm9(args.seed))
