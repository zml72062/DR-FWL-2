"""
script to train on ZINC task
"""

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
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"




def main():
    parser = train_utils.args_setup()
    parser.add_argument('--dataset_name', type=str, default="ZINC", help='name of dataset')
    parser.add_argument('--runs', type=int, default=4, help='number of repeat run')
    args = parser.parse_args()
    args = train_utils.update_args(args)


    path, pre_transform, follow_batch = train_utils.data_setup(args)

    train_dataset = ZINC(path,
                         subset=True,
                         split="train",
                         pre_transform=pre_transform,
                         transform=train_utils.PostTransform(args.wo_node_feature, args.wo_edge_feature))

    val_dataset = ZINC(path,
                       subset=True,
                       split="val",
                       pre_transform=pre_transform,
                       transform=train_utils.PostTransform(args.wo_node_feature, args.wo_edge_feature))

    test_dataset = ZINC(path,
                        subset=True,
                        split="test",
                        pre_transform=pre_transform,
                        transform=train_utils.PostTransform(args.wo_node_feature, args.wo_edge_feature))



    for i in range(1, args.runs + 1):
        logger = WandbLogger(name=f'run_{str(i)}', project=args.exp_name, log_model=True, save_dir=args.save_dir)
        logger.log_hyperparams(args)
        timer = Timer(duration=dict(weeks=4))

        # Set random seed
        seed = train_utils.get_seed(args.seed)
        pl.seed_everything(seed)

        datamodule = PlPyGDataTestonValModule(train_dataset=train_dataset,
                                              val_dataset=val_dataset,
                                              test_dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              follow_batch=follow_batch)
        loss_cri = nn.L1Loss()
        evaluator = torchmetrics.MeanAbsoluteError()

        #TODO: revise pl model module.
        modelmodule = PlGNNTestonValModule(loss_criterion=loss_cri,
                                           evaluator=evaluator,
                                           args=args)

        trainer = Trainer(
                        accelerator="auto",
                        devices="auto",
                        max_epochs=args.num_epochs,
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
        val_result, test_result = trainer.validate(modelmodule, datamodule, ckpt_path="best")
        results = {"final/best_val_metric": val_result["val/metric"],
                   "final/best_test_metric": test_result["test/metric"],
                   "final/avg_train_time_epoch": timer.time_elapsed("train") / args.num_epochs,
                   }
        logger.log_metrics(results)
        wandb.finish()

    return


if __name__ == "__main__":
    main()
