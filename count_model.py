"""
Script to train structure counting dataset collected in I2GNN paper.
"""

import torch
import torch.nn as nn
from counting_dataset import get_count_dataset
import train_utils
from data_utils.preprocess import drfwl2_transform
from torch_geometric.seed import seed_everything
import argparse
from data_utils.batch import collate
import train
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.pool import NodeLevelPooling
from models.gnn_count import DR2FWL2Kernel
from pygmmpp.data import DataLoader

# os.environ["CUDA_LAUNCH_BLOCKING"]="1"


class CountModel(nn.Module):
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

        self.pool = NodeLevelPooling()

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


        x = self.pool(*edge_attrs, *edge_indices, batch.num_nodes)
        x = self.post_mlp(x).squeeze()
        return x

"""
Definition for command-line arguments.
"""
parser = argparse.ArgumentParser()
parser.add_argument('--config-path', type=str, default='configs/count.json',
                    help='Path of the configure file.')
parser.add_argument('--save-dir', type=str, default='results/count',
                    help='Directory to save the result.')
parser.add_argument('--log-file', type=str, default='result.txt',
                    help='Log file name.')
parser.add_argument('--copy-data', action='store_true',
                    help='Whether to copy raw data to result directory.')
parser.add_argument('--runs', type=int, default=3, help='number of repeat run')

args = parser.parse_args()

def train_on_count(seed):
    # Load configure file.
    additional_args = train_utils.load_json(args.config_path)
    loader = train_utils.json_loader(additional_args)

    # Copy necessary info for reproducing result.
    if args.copy_data:
        dir = train_utils.copy(args.config_path, args.save_dir, True, loader.dataset.root)
        root = dir
    else:
        dir = train_utils.copy(args.config_path, args.save_dir)
        root = loader.dataset.root
    
    seed_everything(seed)

    train_dataset = get_count_dataset(root, loader.dataset.target,
                                      split='train',
                                      pre_transform=drfwl2_transform())
    val_dataset = get_count_dataset(root, loader.dataset.target,
                                    split='val',
                                    pre_transform=drfwl2_transform())
    test_dataset = get_count_dataset(root, loader.dataset.target,
                                     split='test',
                                     pre_transform=drfwl2_transform())

    train_val = torch.cat([train_dataset.data_batch.__dict__[loader.dataset.target],
                           val_dataset.data_batch.__dict__[loader.dataset.target]]).to(torch.float)
    mean = train_val.mean(dim=0)
    std = train_val.std(dim=0)

    train_dataset.data_batch.__dict__[loader.dataset.target] = (
                                                                       train_dataset.data_batch.__dict__[
                                                                           loader.dataset.target] - mean
                                                               ) / std
    val_dataset.data_batch.__dict__[loader.dataset.target] = (
                                                                     val_dataset.data_batch.__dict__[
                                                                         loader.dataset.target] - mean
                                                             ) / std
    test_dataset.data_batch.__dict__[loader.dataset.target] = (
                                                                      test_dataset.data_batch.__dict__[
                                                                          loader.dataset.target] - mean
                                                              ) / std

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
    model = CountModel(
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
                        loader.model.dropout)

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
    truth_fn = lambda batch: batch.__dict__[loader.dataset.target]
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
              'min',
              open(args.log_file, 'w'))



if __name__ == "__main__":
    print(f"Use {args.config_path}")
    print(train_on_count(42))
    print(train_on_count(1749))
    print(train_on_count(437))
