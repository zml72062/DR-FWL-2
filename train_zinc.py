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
from data_utils.preprocess import drfwl2_transform
from models.pool import GraphLevelPooling
from models.GNNs import DR2FWL2Kernel
from models.auxiliaries import TwoComponentEmbedding
from pygmmpp.nn.model import MLP

# os.environ["CUDA_LAUNCH_BLOCKING"]="1"


class ZINCModel(nn.Module):
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
        self.lin = TwoComponentEmbedding(21, in_channels)
        self.edge_lin = nn.Embedding(4, in_channels)
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
        self.pool = GraphLevelPooling()
        self.post_mlp = MLP(in_channels, hidden_channels,
                            post_mlp_num_layers, 1, dropout,
                            residual=residual, norm=norm)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.ker.reset_parameters()
        self.post_mlp.reset_parameters()

    def forward(self, batch) -> torch.Tensor:
        x, edge_attr, triangle_1_1_1, triangle_1_1_2, triangle_1_2_2, triangle_2_2_2, \
        inverse_edge_1, inverse_edge_2, edge_index, edge_index2, num_nodes, batch = \
        batch.x, batch.edge_attr, batch.triangle_1_1_1, batch.triangle_1_1_2, batch.triangle_1_2_2, \
        batch.triangle_2_2_2, batch.inverse_edge_1, batch.inverse_edge_2, \
        batch.edge_index, batch.edge_index2, batch.num_nodes, batch.batch0
        x = x.squeeze()
        x, x2 = self.lin(x, x)
        edge_emb = self.edge_lin(edge_attr)
        edge_attr = edge_emb + x[edge_index[0]] + x[edge_index[1]]
        edge_attr2 = x2[edge_index2[0]] + x2[edge_index2[1]]


        edge_attr, edge_attr2 = self.ker(edge_attr,
                                         edge_attr2,
                                         triangle_1_1_1,
                                         triangle_1_1_2,
                                         triangle_1_2_2,
                                         triangle_2_2_2,
                                         inverse_edge_1,
                                         inverse_edge_2)
        x = self.pool(edge_attr, edge_attr2,
                      edge_index, edge_index2, num_nodes, batch)
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
                         pre_transform=drfwl2_transform())

    val_dataset = ZINC(root,
                       subset=True,
                       split="val",
                       pre_transform=drfwl2_transform())

    test_dataset = ZINC(root,
                        subset=True,
                        split="test",
                        pre_transform=drfwl2_transform())

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
        evaluator = torchmetrics.MeanAbsoluteError()
        truth_fn = lambda batch: batch.__dict__[loader.dataset.target]

        """
        Get the model.
        """
        model = ZINCModel(loader.model.in_channels,
                           loader.model.hidden_channels,
                           loader.model.num_layers,
                           None,
                           loader.model.dropout,
                           loader.model.norm,
                           loader.model.mlp_num_layers,
                           loader.model.eps,
                           loader.model.eps2,
                           loader.model.train_eps,
                           loader.model.post_mlp_num_layers)

        #TODO: revise pl model module.
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
