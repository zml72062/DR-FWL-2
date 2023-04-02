"""
Utils file for training.
"""

import argparse
import os
import shutil
import time
import torch
import yaml
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
from typing import Callable


def args_setup():
    r"""Setup argparser.
    """
    parser = argparse.ArgumentParser(f'arguments for training and testing')
    # common args
    parser.add_argument('--save_dir', type=str, default='./save', help='Base directory for saving information.')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Additional configuration file for different dataset and models.')
    parser.add_argument('--seed', type=int, default=234, help='Random seed for reproducibility.')

    #training args
    parser.add_argument('--drop_prob', type=float, default=0.0,
                        help='Probability of zeroing an activation in dropout models.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU. Scales automatically when \
                            multiple GPUs are available.')
    parser.add_argument('--num_workers', type=int, default=4, help='number of worker.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate.')
    parser.add_argument('--l2_wd', type=float, default=0., help='L2 weight decay.')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs.')
    parser.add_argument('--test_eval_interval', type=int, default=10,
                        help='Interval between validation on test dataset.')
    parser.add_argument('--factor', type=float, default=0.5,
                        help='factor in the ReduceLROnPlateau learning rate scheduler.')
    parser.add_argument('--patience', type=int, default=20,
                        help='patience in the ReduceLROnPlateau learning rate scheduler.')

    # data args
    parser.add_argument('--reprocess', action="store_true", help='Whether to reprocess the dataset')


    # model args
    parser.add_argument('--gnn_name', type=str, default="GINE",
                        choices=("GINE", "GINETuple", "resGatedGCN", "gatedGraph"),
                        help='Name of base gnn encoder.')
    parser.add_argument('--model_name', type=str, default="SLFWL",
                        choices=("WL", "KHOP", "SWL", "PSWL", "GSWL", "SSWL", "SSWL+", "LFWL", "SLFWL"),
                        help='Name of GNN model.')
    parser.add_argument('--policy', type=str, choices=("node_deleted", "node_marked", "null",
                        "ego_nets", "ego_nets_plus", "ego_nets_de", "original"), default="ego_nets_de",
                        help="Subgraph Selection policies.")
    parser.add_argument('--num_hops', type=int, default=3, help="Number of hop in ego-net policies.")
    parser.add_argument("--hidden_channels", type=int, default=96, help="Hidden size of the model.")
    parser.add_argument('--wo_node_feature', action='store_true',
                        help='If true, remove node feature from model.')
    parser.add_argument('--wo_edge_feature', action='store_true',
                        help='If true, remove edge feature from model.')
    parser.add_argument("--edge_dim", type=int, default=0, help="Number of edge type.")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layer for GNN.")
    parser.add_argument("--JK", type=str, default="concat",
                        choices=("sum", "max", "mean", "attention", "last", "concat"), help="Jumping knowledge method.")
    parser.add_argument("--residual", action="store_true", help="If ture, use residual connection between each layer.")
    parser.add_argument("--eps", type=float, default=0., help="Initial epsilon in GIN.")
    parser.add_argument("--train_eps", action="store_true", help="If true, the epsilon is trainable.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads in attention.")
    parser.add_argument("--subgraph_pooling", type=str, default="SV", choices=("SV", "VS"),
                        help="Type of pooling in subgraph GNN.")
    parser.add_argument("--pooling_method", type=str, default="mean", choices=("mean", "sum", "attention"),
                        help="Pooling method in graph level tasks.")
    parser.add_argument('--norm_type', type=str, default="Batch",
                        choices=("Batch", "Layer", "Instance", "GraphSize", "Pair", "None"),
                        help="Normalization method in model.")
    return parser


def get_exp_name(args: argparse.ArgumentParser) -> str:
    """Get experiment name.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
    """

    arg_list = []
    if "task" in args:
        arg_list = [str(args.task)]
    arg_list.extend([args.dataset_name,
                      args.gnn_name,
                      args.model_name,
                      str(args.num_layers),
                      str(args.hidden_channels)])

    if args.model_name in ["SWL", "PSWL", "GSWL", "SSWL", "SSWL+", "LFWL", "SLFWL"]:
        arg_list.append(args.policy)
        if args.policy in ["ego_nets", "ego_nets_plus", "ego_nets_de"]:
            arg_list.append(str(args.num_hops))

    if args.residual:
        arg_list.append("residual")

    exp_name = "_".join(arg_list)
    return exp_name + f"-{time.strftime('%Y%m%d%H%M%S')}"


def update_args(args: argparse.ArgumentParser) -> argparse.ArgumentParser:
    r"""Update argparser given config file.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
    """

    if args.config_file is not None:
        with open(args.config_file) as f:
            cfg = yaml.safe_load(f)
        for key, value in cfg.items():
            if isinstance(value, list):
                for v in value:
                    getattr(args, key, []).append(v)
            else:
                setattr(args, key, value)

    args.exp_name = get_exp_name(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    return args

#TODO: Add dataset processing function.
'''
def data_setup(args: argparse.ArgumentParser) -> (str, Callable, list):
    """Setup data for experiment.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
    """
    path_arg_list = [f"data/{args.dataset_name}"]
    follow_batch = []
    graph_feature_compute = lambda x: x

    if args.model_name != "WL":
        path_arg_list.append(args.policy)
        if args.policy in ["ego_nets", "ego_nets_plus", "ego_nets_de"]:
            path_arg_list.append(str(args.num_hops))

        pre_transform = data_utils.policy2transform(args.policy,
                                         num_hops=args.num_hops,
                                         process_subgraphs=graph_feature_compute
                                         )
        follow_batch = ["subgraph_idx", "node_idx"]


    path = "_".join(path_arg_list)
    if os.path.exists(path) and args.reprocess:
        shutil.rmtree(path)

    return path, pre_transform, follow_batch

'''



class PostTransform(object):
    r"""Post transformation of dataset.
    Args:
        wo_node_feature (bool): If true, remove path encoding from model
        wo_edge_feature (bool): If true, remove edge feature from model
    """
    def __init__(self,
                 wo_node_feature: bool,
                 wo_edge_feature: bool,
                 task: int = None):
        self.wo_node_feature = wo_node_feature
        self.wo_edge_feature = wo_edge_feature
        self.task = task

    def __call__(self,
                 data: Data) -> Data:
        if "x" not in data:
            data.x = torch.zeros([data.num_nodes, 1], dtype=torch.float)

        if self.wo_edge_feature:
            data.edge_attr = None
        if self.wo_node_feature:
            data.x = torch.zeros_like(data.x)
        if self.task is not None:
            data.y = data.y[:, self.task]
        return data



def k_fold(dataset, folds, seed):
    r"""Dataset split for K-fold cross-validation.
    Args:
        dataset (Dataset): The dataset to be split.
        folds (int): Number of folds.
        seed (int): Random seed.
    """
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y[dataset.indices()]):
        test_indices.append(torch.from_numpy(idx).long())

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset)).long()
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def get_seed(seed=234):
    r"""Return random seed based on current time.
    Args:
        seed (int): base seed.
    """
    t = int(time.time() * 1000.0)
    seed = seed + ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >> 8) + ((t & 0x0000ff00) << 8) + ((t & 0x000000ff) << 24)
    return seed

