"""
Script to train structure counting dataset collected in I2GNN paper.
"""

import torch
import torch.nn as nn
from counting_dataset import get_count_dataset
import train_utils
from data_utils.preprocess import drfwl2_transform
import pytorch_lightning as pl
from interfaces.pl_model_interface import PlGNNTestonValModule
from interfaces.pl_data_interface import PlPyGDataTestonValModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Timer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import wandb
from torchmetrics import MeanAbsoluteError
import argparse
from models.pool import NodeLevelPooling
from models.GNNs import DR2FWL2Kernel
from pygmmpp.nn.model import MLP

# os.environ["CUDA_LAUNCH_BLOCKING"]="1"


class CountModel(nn.Module):
    def __init__(self,
                 hidden_channels: int,
                 num_layers: int,
                 add_112: bool = True,
                 add_212: bool = True,
                 add_222: bool = True,
                 add_vv: bool = False,
                 eps: float = 0.,
                 norm_type: str = "batch_norm",
                 residual: str = "none",
                 drop_prob: float = 0.0):

        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.add_112 = add_112
        self.add_212 = add_212
        self.add_222 = add_222
        self.add_vv = add_vv
        self.initial_eps = eps
        self.norm_type = norm_type
        self.residual = residual
        self.drop_prob = drop_prob

        self.initial_proj = nn.Linear(1, hidden_channels)
        self.distance_encoding = nn.Embedding(2, hidden_channels)

        self.ker = DR2FWL2Kernel(self.hidden_channels,
                                 self.num_layers,
                                 self.add_112,
                                 self.add_212,
                                 self.add_222,
                                 self.add_vv,
                                 self.initial_eps,
                                 self.norm_type,
                                 self.residual,
                                 self.drop_prob)

        self.pool = NodeLevelPooling()

        self.post_mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels // 2),
                                       nn.ELU(),
                                       nn.Linear(hidden_channels // 2, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.initial_proj.reset_parameters()
        self.distance_encoding.reset_parameters()

        self.ker.reset_parameters()
        for m in self.post_mlp:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def forward(self, batch) -> torch.Tensor:
        x, triangle_0_1_1, triangle_1_1_1, triangle_1_1_2, triangle_1_2_2, triangle_2_2_2, \
        inverse_edge_1, inverse_edge_2, edge_index0, edge_index, edge_index2, num_nodes = \
        batch.x, batch.triangle_0_1_1, \
        batch.triangle_1_1_1, batch.triangle_1_1_2, batch.triangle_1_2_2, \
        batch.triangle_2_2_2, batch.inverse_edge_1, batch.inverse_edge_2, \
        batch.edge_index0, batch.edge_index, batch.edge_index2, batch.num_nodes

        x = self.initial_proj(x)

        edge_attr1 = self.distance_encoding(torch.zeros_like(edge_index[0]))
        edge_attr2 = self.distance_encoding(torch.ones_like(edge_index2[0]))

        edge_attr0 = x
        edge_attr1 = edge_attr1 + x[edge_index[1]]
        edge_attr2 = edge_attr2 + x[edge_index2[1]]


        edge_attr0, edge_attr1, edge_attr2 = self.ker(edge_attr0,
                                                      edge_attr1,
                                                      edge_attr2,
                                                      edge_index0,
                                                      edge_index,
                                                      edge_index2,
                                                      triangle_0_1_1,
                                                      triangle_1_1_1,
                                                      triangle_1_1_2,
                                                      triangle_1_2_2,
                                                      triangle_2_2_2,
                                                      inverse_edge_1,
                                                      inverse_edge_2)
        x = self.pool(edge_attr0, edge_attr1, edge_attr2, edge_index, edge_index2, num_nodes)
        x = self.post_mlp(x).squeeze()
        return x


def main():
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

    exp_name = train_utils.get_exp_name(loader)
    for i in range(1, args.runs + 1):
        logger = WandbLogger(name=f'run_{str(i)}', project=exp_name, log_model=True, save_dir=root)
        logger.log_hyperparams(additional_args)
        timer = Timer(duration=dict(weeks=4))


        # Set random seed
        seed = train_utils.get_seed(loader.train.seed)
        pl.seed_everything(seed)

        datamodule = PlPyGDataTestonValModule(train_dataset=train_dataset,
                                              val_dataset=val_dataset,
                                              test_dataset=test_dataset,
                                              batch_size=loader.train.batch_size,
                                              num_workers=loader.train.num_workers)
        loss_cri = nn.L1Loss()
        evaluator = MeanAbsoluteError()
        truth_fn = lambda batch: batch.__dict__[loader.dataset.target]

        """
        Get the model.
        """
        model = CountModel(
                           loader.model.hidden_channels,
                           loader.model.num_layers,
                           loader.model.add_112,
                           loader.model.add_212,
                           loader.model.add_222,
                           loader.model.add_vv,
                           loader.model.eps,
                           loader.model.norm,
                           loader.model.residual,
                           loader.model.dropout)


        modelmodule = PlGNNTestonValModule(model=model,
                                           loss_criterion=loss_cri,
                                           evaluator=evaluator,
                                           truth_fn=truth_fn,
                                           loader=loader
                                           )
        trainer = Trainer(
                        accelerator="auto",
                        devices="auto",
                        max_epochs=loader.train.epochs,
                        enable_checkpointing=True,
                        enable_progress_bar=True,
                        logger=logger,
                        callbacks=[
                            TQDMProgressBar(refresh_rate=20),
                            ModelCheckpoint(monitor="val/metric", mode="min"),
                            LearningRateMonitor(logging_interval="epoch"),
                            timer
                        ]
                        )


        trainer.fit(modelmodule, datamodule)
        modelmodule.set_test_eval_still()
        val_result, test_result = trainer.validate(modelmodule, datamodule)
        results = {"final/best_val_metric": val_result["val/metric"],
                   "final/best_test_metric": test_result["test/metric"],
                   "final/avg_train_time_epoch": timer.time_elapsed("train") / loader.train.epochs,
                   }
        logger.log_metrics(results)
        wandb.finish()

    return


if __name__ == "__main__":
    main()

