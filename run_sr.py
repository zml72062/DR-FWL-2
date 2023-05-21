"""
script to train on SR classification dataset
"""
import argparse
import os
import shutil
import time
from json import dumps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DataParallel
from torch_geometric.seed import seed_everything

from models.gnn_count import DR2FWL2Kernel
from pygmmpp.datasets import SRDataset
from data_utils.batch import collate
from pygmmpp.data import DataLoader
from data_utils.preprocess import drfwl2_transform
from models.pool import GraphLevelPooling
from pygmmpp.utils import compose


def train(loader, model, optimizer, device, parallel=False):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        if parallel:
            num_graphs = len(data)
            y = torch.cat([d.y for d in data]).to(device)
        else:
            num_graphs = data.num_graphs
            data = data.to(device)
            y = data.y
        out = model(data).squeeze()
        loss = torch.nn.NLLLoss()(out, y)
        loss.backward()
        total_loss += loss.item() * num_graphs
        optimizer.step()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(loader, model, device, parallel=False):
    model.train()  # eliminate the effect of BN
    y_preds, y_trues = [], []
    for data in loader:
        if parallel:
            y = torch.cat([d.y for d in data]).to(device)
        else:
            data = data.to(device)
            y = data.y
        y_preds.append(torch.argmax(model(data), dim=-1))
        y_trues.append(y)
    y_preds = torch.cat(y_preds, -1)
    y_trues = torch.cat(y_trues, -1)
    return (y_preds == y_trues).float().mean()

class SRModel(nn.Module):
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

        self.node_transform = nn.Linear(1, self.hidden_channels)

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
                                       nn.Linear(hidden_channels // 2, 15))
        
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


def main():
    parser = argparse.ArgumentParser(f'arguments for training and testing')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--root', type=str, default='datasets/sr25')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layer', type=int, default=5)
    parser.add_argument('--cuda', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2-wd', type=float, default=0.0)
    parser.add_argument('--num-epochs', type=int, default=100)

    args = parser.parse_args()
    seed_everything(args.seed)

    dataset = SRDataset(args.root, pre_transform=drfwl2_transform())
    dataset.data_batch.y = torch.arange(len(dataset.data_batch.y)).long()  # each graph is a unique class
    train_dataset = dataset
    val_dataset = dataset
    test_dataset = dataset

    # 2. create loader
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collator=collate)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, collator=collate)

    device = f'cuda:{args.cuda}' if args.cuda != -1 else 'cpu'

    model = SRModel(
        args.hidden,
        args.layer,
        norm_type='none',
        norm_between_layers='none',
        residual='last'
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_wd)
    best_test_acc = 0
    start_outer = time.time()
    for epoch in range(args.num_epochs):
        start = time.time()
        train_loss = train(train_loader, model, optimizer, device=device)
        lr = optimizer.param_groups[0]['lr']
        test_acc = test(test_loader, model, device=device)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
        time_per_epoch = time.time() - start

        print(f'Epoch: {epoch + 1:03d}, LR: {lr:7f}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}, '
                 f'Best Test Acc: {best_test_acc:.4f}, Seconds: {time_per_epoch:.4f}')
        torch.cuda.empty_cache()  # empty test part memory cost

    time_average_epoch = time.time() - start_outer
    print(
        f'Loss: {train_loss:.4f}, Best test: {best_test_acc:.4f}, Seconds/epoch: {time_average_epoch / (epoch + 1):.4f}')


if __name__ == "__main__":
    main()
