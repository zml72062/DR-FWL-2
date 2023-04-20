"""
script to train on ZINC task
"""
import torch
import torch.nn as nn
from pygmmpp.datasets import ZINC
import train_utils
import pytorch_lightning as pl
from interfaces.pl_model_interface import PlGNNTestonValModule
from interfaces.pl_data_interface import PlPyGDataTestonValModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Timer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torchmetrics
import wandb
import argparse
from data_utils.preprocess import drfwl2_transform_zinc
from models.pool import GraphLevelPooling
from models.GNNs import DR2FWL2KernelZINC
from models.utils import clones
from pygmmpp.nn.model import MLP

# os.environ["CUDA_LAUNCH_BLOCKING"]="1"


class ZINCModel(nn.Module):
    def __init__(self,
                 hidden_channels: int,
                 num_layers: int,
                 add_0: bool = True,
                 add_112: bool = True,
                 add_212: bool = True,
                 add_222: bool = True,
                 add_321: bool = True,
                 add_331: bool = True,
                 add_vv: bool = False,
                 eps: float = 0.,
                 train_eps: bool = False,
                 norm_type: str = "batch_norm",
                 residual: str = "none",
                 drop_prob: float = 0.0):

        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.add_0 = add_0
        self.add_112 = add_112
        self.add_212 = add_212
        self.add_222 = add_222
        self.add_321 = add_321
        self.add_331 = add_331
        self.add_vv = add_vv
        self.initial_eps = eps
        self.train_eps = train_eps
        self.norm_type = norm_type
        self.residual = residual
        self.drop_prob = drop_prob

        self.initial_proj = nn.Embedding(21, hidden_channels)
        self.distance_encoding = nn.Embedding(3, hidden_channels)
        edge_lin = nn.Embedding(4, hidden_channels)
        self.edge_lins = clones(edge_lin, hidden_channels)

        self.ker = DR2FWL2KernelZINC(self.hidden_channels,
                                     self.num_layers,
                                     self.add_0,
                                     self.add_112,
                                     self.add_212,
                                     self.add_222,
                                     self.add_321,
                                     self.add_331,
                                     self.add_vv,
                                     self.initial_eps,
                                     self.train_eps,
                                     self.norm_type,
                                     self.residual,
                                     self.drop_prob)

        self.pool = GraphLevelPooling()

        self.post_mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels // 2),
                                       nn.ELU(),
                                       nn.Linear(hidden_channels // 2, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.initial_proj.reset_parameters()
        self.distance_encoding.reset_parameters()
        for e in self.edge_lins:
            e.reset_parameters()
        self.ker.reset_parameters()
        for m in self.post_mlp:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def forward(self, batch) -> torch.Tensor:
        x, edge_feature, triangle_0_1_1, triangle_1_1_1, triangle_1_1_2, triangle_1_2_2, triangle_2_2_2, triangle_3_2_1, triangle_3_3_1, \
        inverse_edge_1, inverse_edge_2, inverse_edge_3, edge_index0, edge_index, edge_index2, edge_index3, num_nodes, batch = \
        batch.x, batch.edge_attr, batch.triangle_0_1_1, \
        batch.triangle_1_1_1, batch.triangle_1_1_2, batch.triangle_1_2_2, \
        batch.triangle_2_2_2, batch.triangle_3_2_1, batch.triangle_3_3_1, batch.inverse_edge_1, batch.inverse_edge_2, batch.inverse_edge_3,\
        batch.edge_index0, batch.edge_index, batch.edge_index2, batch.edge_index3, batch.num_nodes, batch.batch0
        x = x.squeeze()
        x = self.initial_proj(x)
        edge_attr1 = self.distance_encoding(torch.zeros_like(edge_index[0]))
        edge_attr2 = self.distance_encoding(torch.ones_like(edge_index2[0]))
        edge_attr3 = self.distance_encoding(torch.ones_like(edge_index3[0]) * 2)


        edge_attr0 = x
        edge_attr1 = edge_attr1 + x[edge_index[1]] + x[edge_index[0]]
        edge_attr2 = edge_attr2 + x[edge_index2[1]] + x[edge_index2[0]]
        edge_attr3 = edge_attr3 + x[edge_index3[1]] + x[edge_index3[0]]


        edge_emb_list = [l(edge_feature) for l in self.edge_lins]

        edge_attr0, edge_attr1, edge_attr2, edge_attr3 = self.ker(edge_attr0,
                                                                  edge_attr1,
                                                                  edge_attr2,
                                                                  edge_attr3,
                                                                  edge_index0,
                                                                  edge_index,
                                                                  edge_index2,
                                                                  edge_index3,
                                                                  triangle_0_1_1,
                                                                  triangle_1_1_1,
                                                                  triangle_1_1_2,
                                                                  triangle_1_2_2,
                                                                  triangle_2_2_2,
                                                                  triangle_3_2_1,
                                                                  triangle_3_3_1,
                                                                  inverse_edge_1,
                                                                  inverse_edge_2,
                                                                  inverse_edge_3,
                                                                  edge_emb_list)
        x = self.pool([edge_attr0, edge_attr1, edge_attr2, edge_attr3],
                      [edge_index0, edge_index, edge_index2, edge_index3],
                      num_nodes, batch)
        x = self.post_mlp(x).squeeze()
        return x


def main():
    """
    Definition for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='configs/zinc.json',
                        help='Path of the configure file.')
    parser.add_argument('--save-dir', type=str, default='results/zinc',
                        help='Directory to save the result.')
    parser.add_argument('--log-file', type=str, default='result.txt',
                        help='Log file name.')
    parser.add_argument('--copy-data', action='store_true',
                        help='Whether to copy raw data to result directory.')
    parser.add_argument('--runs', type=int, default=10, help='number of repeat run')
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

    train_dataset = ZINC(root,
                         subset=True,
                         split="train",
                         pre_transform=drfwl2_transform_zinc())

    val_dataset = ZINC(root,
                       subset=True,
                       split="val",
                       pre_transform=drfwl2_transform_zinc())

    test_dataset = ZINC(root,
                        subset=True,
                        split="test",
                        pre_transform=drfwl2_transform_zinc())

    exp_name = train_utils.get_exp_name(loader)
    for i in range(1, args.runs + 1):
        logger = WandbLogger(name=f'run_{str(i)}', project=exp_name, log_model=True, save_dir=args.save_dir)
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
        evaluator = torchmetrics.MeanAbsoluteError()
        truth_fn = lambda batch: batch.__dict__[loader.dataset.target]

        """
        Get the model.
        """
        model = ZINCModel(
                           loader.model.hidden_channels,
                           loader.model.num_layers,
                           loader.model.add_0,
                           loader.model.add_112,
                           loader.model.add_212,
                           loader.model.add_222,
                           loader.model.add_321,
                           loader.model.add_331,
                           loader.model.add_vv,
                           loader.model.eps,
                           loader.model.train_eps,
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
