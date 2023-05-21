import torch
import torch.nn as nn
import torch.nn.functional
from HomologyTAPEDatasetWithCount import HomologyTAPEDatasetWithCount
from ProteinsDBDatasetWithCount import ProteinsDBDatasetWithCount
from ProtFunctDatasetWithCount import ProtFunctDatasetWithCount
from PygHomologyTAPEDatasetWithCount import PygHomologyTAPEDatasetWithCount
from PygProteinsDBDatasetWithCount import PygProteinsDBDatasetWithCount
from PygProtFunctDatasetWithCount import PygProtFunctDatasetWithCount
from data_utils.preprocess import drfwl2_transform
from torch_geometric.seed import seed_everything
import argparse
from tqdm import tqdm
from count_I2GNN import (GNN as MPNNCounting, 
                         I2GNN as I2GNNCounting, 
                         NGNN as NGNNCounting, 
                         PPGN as PPGNCounting)
from data_utils.batch import collate
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.pool import NodeLevelPooling
from models.gnn_count import DR2FWL2Kernel
from utils_i2 import create_subgraphs, create_subgraphs2
from pygmmpp.data import DataLoader as myDataLoader
from dataloader import DataLoader as pyDataLoader
import sys
import time
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"

target_map = {
    '3-cycle': 1,
    '4-cycle': 2,
    '5-cycle': 3,
    '6-cycle': 4,
    '4-path': 5
}

class DRFWL2Counting(nn.Module):
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
                 residual: str = "last",
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
parser.add_argument('--dataset', type=str, default='HomologyTAPE',
                    help='ProteinsDB/HomologyTAPE/ProtFunct')
parser.add_argument('--model', type=str, default='DRFWL2',
                    help='MPNN/NGNN/I2GNN/DRFWL2/PPGN')
parser.add_argument('--seed', type=int, default=42, help='random seed.')
parser.add_argument('--root', type=str, default='homology')
parser.add_argument('--target', type=str, default='6-cycle',
                    help='3/4/5/6-cycle/4-path')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_patience', type=int, default=10)
parser.add_argument('--lr_decay', type=float, default=0.9)
parser.add_argument('--lr_min', type=float, default=1e-5)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--inference', action='store_true', default=False)

# Command-line arguments for only NGNN and I2GNN

parser.add_argument('--h', type=int, default=None, help='hop of enclosing subgraph;\
                    if None, will not use NestedGNN')
parser.add_argument('--max_nodes_per_hop', type=int, default=None)
parser.add_argument('--node_label', type=str, default='hop',
                    help='apply distance encoding to nodes within each subgraph, use node\
                    labels as additional node features; support "hop", "drnl", "spd", \
                    for "spd", you can specify number of spd to keep by "spd3", "spd4", \
                    "spd5", etc. Default "spd"=="spd2".')
parser.add_argument('--use_rd', action='store_true', default=False, 
                    help='use resistance distance as additional node labels')

# Command-line arguments for only ProteinsDBDataset

parser.add_argument('--test_split', type=int, default=0, help='0-9')

args = parser.parse_args()

def get_transform():
    if args.model == 'DRFWL2':
        return drfwl2_transform()
    elif args.model == 'NGNN':
        return lambda g: create_subgraphs(g, args.h,
                                max_nodes_per_hop=args.max_nodes_per_hop, 
                                node_label=args.node_label, 
                                use_rd=args.use_rd,
                                save_relabel=True)
    elif args.model == 'I2GNN':
        return lambda g: create_subgraphs2(g, args.h,
                                 max_nodes_per_hop=args.max_nodes_per_hop,
                                 node_label=args.node_label,
                                 use_rd=args.use_rd,
                                 )     
    elif args.model in {'MPNN', 'PPGN'}:
        return None

def epoch(model, 
          loader, 
          device, 
          optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()
    
    loss, dataset_len = 0.0, 0
    for batch in loader:
        if optimizer is not None:
            optimizer.zero_grad()
        batch = batch.to(device)
        truth = batch.__dict__[args.target] if not args.model in {'NGNN', 'I2GNN'} else batch.y[:, target_map[args.target]-1]
        pred = model(batch).squeeze()
        batch_loss = nn.L1Loss()(pred, truth)

        if optimizer is not None:
            batch_loss.backward()
            optimizer.step()

        with torch.no_grad():
            loss += batch_loss.item() * batch.num_graphs
            dataset_len += batch.num_graphs
    return loss/dataset_len

def train_on_count():
    seed_everything(args.seed)

    if args.model in {'NGNN', 'I2GNN'}:
        dataset = eval(f"Pyg{args.dataset}DatasetWithCount")
        DataLoader = pyDataLoader
        dataloader_kwargs = {}
    else:
        dataset = eval(f"{args.dataset}DatasetWithCount")
        DataLoader = myDataLoader
        dataloader_kwargs = {}
        if args.model == 'DRFWL2':
            dataloader_kwargs = {'collator': collate}
    print(f"Use {args.dataset} dataset, {args.model} model")

    start_preprocess = time.time()
    if args.dataset == 'ProteinsDB':
        datasets = [dataset(args.root, i, includeHB=True, pre_transform=get_transform())
                    for i in range(10)]
        test_split, valid_split = args.test_split, (args.test_split + 1) % 10
        train_splits = [i for i in range(10) if i != test_split and i != valid_split]
        train_val_splits = train_splits + [valid_split]
    else:
        train_dataset = dataset(args.root, 'training', includeHB=True, 
                                pre_transform=get_transform())
        valid_dataset = dataset(args.root, 'validation', includeHB=True, 
                                pre_transform=get_transform())
            
        if args.dataset == 'HomologyTAPE':
            test_fold_dataset = dataset(args.root, 'test_fold', includeHB=True,
                                        pre_transform=get_transform())
            test_family_dataset = dataset(args.root, 'test_family', includeHB=True,
                                        pre_transform=get_transform())      
            test_superfamily_dataset = dataset(args.root, 'test_superfamily', includeHB=True,
                                            pre_transform=get_transform())
        elif args.dataset == 'ProtFunct':
            test_dataset = dataset(args.root, 'testing', includeHB=True,
                                pre_transform=get_transform())
    end_preprocess = time.time()
    print("Pre-processing time:", end_preprocess - start_preprocess)
    if args.dataset == 'ProteinsDB':
        if args.model in {'NGNN', 'I2GNN'}:
            train_val = torch.cat([datasets[i].data.y[:, target_map[args.target]-1] for i in train_val_splits]).to(torch.float)
            mean = train_val.mean(dim=0)
            std = train_val.std(dim=0)
            print(f"Mean: {mean},   Std: {std}")
            for i in range(10):
                datasets[i].data.y = (datasets[i].data.y.to(torch.float) - mean) / std
        else:
            train_val = torch.cat([datasets[i].data_batch.__dict__[args.target] for i in train_val_splits]).to(torch.float)
            mean = train_val.mean(dim=0)
            std = train_val.std(dim=0)
            print(f"Mean: {mean},   Std: {std}")
            for i in range(10):
                datasets[i].data_batch.__dict__[args.target] = (datasets[i].data_batch.__dict__[args.target].to(torch.float) - mean) / std
    else:  
        if args.model in {'NGNN', 'I2GNN'}:
            train_val = torch.cat([train_dataset.data.y[:, target_map[args.target]-1],
                                valid_dataset.data.y[:, target_map[args.target]-1]]).to(torch.float)
            mean = train_val.mean(dim=0)
            std = train_val.std(dim=0)
            print(f"Mean: {mean},   Std: {std}")
            train_dataset.data.y = (train_dataset.data.y.to(torch.float) - mean) / std
            valid_dataset.data.y = (valid_dataset.data.y.to(torch.float) - mean) / std
            if args.dataset == 'HomologyTAPE':
                test_fold_dataset.data.y = (test_fold_dataset.data.y.to(torch.float) - mean) / std
                test_family_dataset.data.y = (test_family_dataset.data.y.to(torch.float) - mean) / std
                test_superfamily_dataset.data.y = (test_superfamily_dataset.data.y.to(torch.float) - mean) / std
            elif args.dataset == 'ProtFunct':
                test_dataset.data.y = (test_dataset.data.y.to(torch.float) - mean) / std
        else:
            train_val = torch.cat([train_dataset.data_batch.__dict__[args.target],
                                valid_dataset.data_batch.__dict__[args.target]]).to(torch.float)
            mean = train_val.mean(dim=0)
            std = train_val.std(dim=0)
            print(f"Mean: {mean},   Std: {std}")

            train_dataset.data_batch.__dict__[args.target] = (train_dataset.data_batch.__dict__[args.target].to(torch.float) - mean) / std
            valid_dataset.data_batch.__dict__[args.target] = (valid_dataset.data_batch.__dict__[args.target].to(torch.float) - mean) / std
            if args.dataset == 'HomologyTAPE':
                test_fold_dataset.data_batch.__dict__[args.target] = (test_fold_dataset.data_batch.__dict__[args.target].to(torch.float) - mean) / std
                test_family_dataset.data_batch.__dict__[args.target] = (test_family_dataset.data_batch.__dict__[args.target].to(torch.float) - mean) / std
                test_superfamily_dataset.data_batch.__dict__[args.target] = (test_superfamily_dataset.data_batch.__dict__[args.target].to(torch.float) - mean) / std
            elif args.dataset == 'ProtFunct':
                test_dataset.data_batch.__dict__[args.target] = (test_dataset.data_batch.__dict__[args.target].to(torch.float) - mean) / std

    """
    Load the dataset.
    """
    if args.dataset == 'ProteinsDB':
        loaders = [DataLoader(dataset, batch_size=args.batch_size,
                              shuffle=i in train_splits, **dataloader_kwargs)
                    for (i, dataset) in enumerate(datasets)]
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=True, **dataloader_kwargs)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                shuffle=False, **dataloader_kwargs)
        if args.dataset == 'HomologyTAPE':
            test_fold_loader = DataLoader(test_fold_dataset, batch_size=args.batch_size,
                                        shuffle=False, **dataloader_kwargs)
            test_family_loader = DataLoader(test_family_dataset, batch_size=args.batch_size,
                                            shuffle=False, **dataloader_kwargs)
            test_superfamily_loader = DataLoader(test_superfamily_dataset, batch_size=args.batch_size,
                                                shuffle=False, **dataloader_kwargs) 
        elif args.dataset == 'ProtFunct':
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                    shuffle=False, **dataloader_kwargs)

    """
    Set the device.
    """
    device = f"cuda:{args.cuda}" if args.cuda != -1 else "cpu"

    """
    Get the model.
    """
    model = eval(f"{args.model}Counting")(hidden_channels=args.hidden_channels,
                                          num_layers=args.num_layers)
    
    print("# of Parameters:", sum([p.numel() for p in model.parameters()]))

    """
    Get the optimizer.
    """
    optimizer = Adam(model.parameters(), lr=args.lr)

    """
    Get the LR scheduler.
    """
    scheduler = ReduceLROnPlateau(optimizer, 'min',
                                  factor=args.lr_decay,
                                  patience=args.lr_patience,
                                  min_lr=args.lr_min)

    """
    Run the training script.
    """
    model.to(device)

    best_val_loss = 1e6
    if args.dataset == 'HomologyTAPE':
        best_test_fold_loss = 0
        best_test_family_loss = 0
        best_test_superfamily_loss = 0
    else:
        best_test_loss = 0

    next_run = time.time()
    for idx in range(args.epochs):
        if args.dataset != 'ProteinsDB':
            if args.inference:
                with torch.no_grad():
                    train_loss = epoch(model, train_loader, device, None)
            else:
                train_loss = epoch(model, train_loader, device, optimizer)
            with torch.no_grad():
                val_loss = epoch(model, val_loader, device, None)
                if args.dataset == 'HomologyTAPE':
                    test_fold_loss = epoch(model, test_fold_loader, device, None)
                    test_family_loss = epoch(model, test_family_loader, device, None)
                    test_superfamily_loss = epoch(model, test_superfamily_loader, device, None)
                elif args.dataset == 'ProtFunct':
                    test_loss = epoch(model, test_loader, device, None)
        else:
            if args.inference:
                with torch.no_grad():
                    train_loss = sum(
                        [epoch(model, loaders[i], device, None) for i in train_splits]
                    ) / 8
            else:
                train_loss = sum(
                    [epoch(model, loaders[i], device, optimizer) for i in train_splits]
                ) / 8
            with torch.no_grad():
                val_loss = epoch(model, loaders[valid_split], device, None)
                test_loss = epoch(model, loaders[test_split], device, None)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.dataset == 'HomologyTAPE':
                best_test_fold_loss = test_fold_loss
                best_test_family_loss = test_family_loss
                best_test_superfamily_loss = test_superfamily_loss
            else:
                best_test_loss = test_loss
        
        if idx % 50 == 49:
            fifty_run = time.time()
            print("Running time for 50 epochs: ", fifty_run - next_run)
            next_run = fifty_run
        scheduler.step(val_loss)

        print("Epoch %d: " % idx)
        print("Training MAE: %f" % train_loss)
        print("Validation MAE: %f" % val_loss)
        if args.dataset == 'HomologyTAPE':
            print("Test Fold MAE: %f" % test_fold_loss)
            print("Test Family MAE: %f" % test_family_loss)
            print("Test Superfamily MAE: %f" % test_superfamily_loss)
        else:
            print("Test MAE: %f" % test_loss)

    print("Best Validation MAE: %f" % best_val_loss)
    if args.dataset == 'HomologyTAPE':
        print("Best Test Fold MAE: %f" % best_test_fold_loss)
        print("Best Test Family MAE: %f" % best_test_family_loss)
        print("Best Test Superfamily MAE: %f" % best_test_superfamily_loss)
    else:
        print("Best Test MAE: %f" % best_test_loss)

if __name__ == "__main__":
    train_on_count()
