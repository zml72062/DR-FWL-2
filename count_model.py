from pygmmpp.utils import compose
from pygmmpp.nn.model import MLP
from pygmmpp.data import DataLoader
from model import (DR2FWL2Kernel, 
                   NodeLevelPooling, 
                   TwoComponentLinear, 
                   TwoComponentReLU)
from counting_dataset import get_count_dataset
from preprocess import (generate_k_hop_neighbor,
                        generate_k_hop_neighbor_feature,
                        generate_inverse_edge,
                        generate_lkm_triangle)
from json_loader import json_loader
from save_result import copy
from batch import collate
from torch_geometric.seed import seed_everything
import argparse
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import train
import os.path as osp

class CountModel(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int,
                 num_layers: int,
                 residual: str = 'cat',
                 dropout: float = 0.0,
                 norm: str = 'batch_norm',
                 mlp_num_layers: int = 1,
                 eps: float = 0.0,
                 eps2: float = 0.0,
                 train_eps: bool = False,
                 post_mlp_num_layers: int = 2):
        super().__init__()
        self.lin = TwoComponentLinear(1, in_channels)
        self.ker = DR2FWL2Kernel(in_channels, 
                                 hidden_channels,
                                 num_layers,
                                 dropout,
                                 residual=residual,
                                 norm=norm,
                                 mlp_num_layers=mlp_num_layers,
                                 mlp_dropout=dropout,
                                 eps=eps,
                                 eps2=eps2,
                                 train_eps=train_eps,
                                 mlp_norm=norm)
        self.pool = NodeLevelPooling()
        self.post_mlp = MLP(in_channels, hidden_channels,
                            post_mlp_num_layers, 1, dropout,
                            residual=residual, norm=norm)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.ker.reset_parameters()
        self.post_mlp.reset_parameters()

    def forward(self,
                edge_attr: torch.Tensor,
                edge_attr2: torch.Tensor,
                triangle_1_1_1: torch.LongTensor,
                triangle_1_1_2: torch.LongTensor,
                triangle_1_2_2: torch.LongTensor,
                triangle_2_2_2: torch.LongTensor,
                inverse_edge_1: torch.LongTensor,
                inverse_edge_2: torch.LongTensor,
                edge_index: torch.LongTensor,
                edge_index2: torch.LongTensor,
                num_nodes: int) -> torch.Tensor:
        edge_attr, edge_attr2 = self.lin(edge_attr, edge_attr2)
        edge_attr, edge_attr2 = TwoComponentReLU()(edge_attr, edge_attr2)
        edge_attr, edge_attr2 = self.ker(edge_attr,
                                         edge_attr2,
                                         triangle_1_1_1,
                                         triangle_1_1_2,
                                         triangle_1_2_2,
                                         triangle_2_2_2,
                                         inverse_edge_1,
                                         inverse_edge_2)
        print(edge_attr, edge_attr2)
        x = self.pool(edge_attr, edge_attr2,
                      edge_index, edge_index2, num_nodes)
        return self.post_mlp(x).squeeze()

pretransform = compose(
   [generate_k_hop_neighbor(2), 
    generate_k_hop_neighbor_feature(2, False),
    generate_lkm_triangle(1, 1, 1),
    generate_lkm_triangle(1, 1, 2),
    generate_lkm_triangle(1, 2, 2),
    generate_lkm_triangle(2, 2, 2),
    generate_inverse_edge(2)]
)

    

if __name__ == '__main__':
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
    args = parser.parse_args()

    """
    Load configure file.
    """
    loader = json_loader(args.config_path)

    """
    Copy necessary info for reproducing result.
    """
    if args.copy_data:
        dir = copy(args.config_path, args.save_dir, True, loader.dataset.root)
        root = dir
    else:
        dir = copy(args.config_path, args.save_dir)
        root = loader.dataset.root

    """
    Set the random seed.
    """
    seed_everything(loader.train.seed)

    """
    Get the dataset and normalize data.
    """
    train_dataset = get_count_dataset(root, loader.dataset.target,
                                      split='train',
                                      pre_transform=pretransform)
    val_dataset = get_count_dataset(root, loader.dataset.target,
                                    split='val',
                                    pre_transform=pretransform)
    test_dataset = get_count_dataset(root, loader.dataset.target,
                                     split='test',
                                     pre_transform=pretransform)
    
    train_val = torch.cat([train_dataset.data_batch.__dict__[loader.dataset.target],
                           val_dataset.data_batch.__dict__[loader.dataset.target]]).to(torch.float)
    mean = train_val.mean(dim=0)
    std = train_val.std(dim=0)

    train_dataset.data_batch.__dict__[loader.dataset.target] = (
        train_dataset.data_batch.__dict__[loader.dataset.target] - mean
    ) / std
    val_dataset.data_batch.__dict__[loader.dataset.target] = (
        val_dataset.data_batch.__dict__[loader.dataset.target] - mean
    ) / std
    test_dataset.data_batch.__dict__[loader.dataset.target] = (
        test_dataset.data_batch.__dict__[loader.dataset.target] - mean
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
    model = CountModel(loader.model.in_channels,
                       loader.model.hidden_channels,
                       loader.model.num_layers,
                       loader.model.residual,
                       loader.model.dropout,
                       loader.model.norm,
                       loader.model.mlp_num_layers,
                       loader.model.eps,
                       loader.model.eps2,
                       loader.model.train_eps,
                       loader.model.post_mlp_num_layers)
    
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
    pred_fn = lambda model, batch: model(batch.edge_attr,
                                         batch.edge_attr2,
                                         batch.triangle_1_1_1,
                                         batch.triangle_1_1_2,
                                         batch.triangle_1_2_2,
                                         batch.triangle_2_2_2,
                                         batch.inverse_edge_1,
                                         batch.inverse_edge_2,
                                         batch.edge_index,
                                         batch.edge_index2,
                                         batch.num_nodes)
    truth_fn = lambda batch: batch.__dict__[loader.dataset.target]
    loss_fn = nn.MSELoss()
    metric = lambda pred, truth: (pred - truth).abs().mean(dim=0)

    """
    Run the training script.
    """
    train.run(loader.train.epochs,
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
              open(osp.join(dir, args.log_file), 'w'))
    