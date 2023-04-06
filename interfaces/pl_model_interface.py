from argparse import ArgumentParser
from copy import deepcopy as c
from typing import Dict, List, Callable
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torchmetrics import Metric


class PlGNNModule(pl.LightningModule):
    """Basic pytorch lighting module for GNNs.
    Args:
        model (nn.Module): Model to be trained or evaluated.
        loss_criterion (nn.Module) : Loss compute module.
        evaluator (Metric): Evaluator for evaluating model performance.
        truth_fn (Callable): A function to get true label from batched_data.
        loader (Dict): Arguments loader.
    """

    def __init__(self,
                 model: nn.Module,
                 loss_criterion: nn.Module,
                 evaluator: Metric,
                 truth_fn: Callable,
                 loader: Dict
                 ):
        super(PlGNNModule, self).__init__()
        self.model = model
        self.loss_criterion = loss_criterion
        self.train_evaluator = c(evaluator)
        self.val_evaluator = c(evaluator)
        self.test_evaluator = c(evaluator)
        self.loader = loader
        self.truth_fn = truth_fn

    def forward(self,
                data: Data) -> Tensor:
        return self.model(data)

    def training_step(self,
                      batch: Data,
                      batch_idx: Tensor) -> Dict:
        y = self.truth_fn(batch)
        out = self.model(batch).squeeze()
        loss = self.loss_criterion(out, y)
        self.log("train/loss",
                 loss,
                 prog_bar=True,
                 batch_size=self.loader.train.batch_size,
                 sync_dist=True)
        return {'loss': loss, 'preds': out, 'target': y}

    def training_step_end(self,
                          outputs: Dict) -> None:
        self.train_evaluator.update(outputs["preds"], outputs["target"])

    def training_epoch_end(self,
                           outputs: List) -> None:
        self.log("train/metric",
                 self.train_evaluator.compute(),
                 on_epoch=True,
                 on_step=False,
                 prog_bar=False)
        self.train_evaluator.reset()

    def validation_step(self,
                        batch: Data,
                        batch_idx: Tensor,
                        data_loader_idx: int) -> Dict:
        y = self.truth_fn(batch)
        out = self.model(batch).squeeze()
        loss = self.loss_criterion(out, y)
        self.log("val/loss",
                 loss,
                 prog_bar=False,
                 batch_size=self.loader.train.batch_size,
                 sync_dist=True)
        return {'loss': loss, 'preds': out, 'target': y}

    def validation_step_end(self,
                            outputs: Dict) -> None:
        self.val_evaluator.update(outputs["preds"], outputs["target"])

    def validation_epoch_end(self,
                             outputs: List) -> None:
        self.log("val/metric",
                 self.val_evaluator.compute(),
                 on_epoch=True,
                 on_step=False,
                 prog_bar=True)
        self.val_evaluator.reset()

    def test_step(self,
                  batch: Data,
                  batch_idx: Tensor) -> Dict:
        y = self.truth_fn(batch)
        out = self.model(batch).squeeze()
        loss = self.loss_criterion(out, y)
        self.log("test/loss",
                 loss,
                 prog_bar=False,
                 batch_size=self.loader.train.batch_size,
                 sync_dist=True)
        return {'loss': loss, 'preds': out, 'target': y}

    def test_step_end(self,
                      outputs: Dict) -> None:
        self.test_evaluator.update(outputs["preds"], outputs["target"])

    def test_epoch_end(self,
                       outputs: List) -> None:
        self.log("test/metric",
                 self.test_evaluator.compute(),
                 on_epoch=True,
                 on_step=False,
                 prog_bar=True)
        self.test_evaluator.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.loader.train.lr,
            weight_decay=self.loader.train.l2_penalty,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=self.loader.train.lr_reduce_factor,
                    patience=self.loader.train.lr_reduce_patience,
                    min_lr=self.loader.train.lr_reduce_min
                ),
                "monitor": "val/metric",
                "frequency": 1,
                "interval": "epoch",
            },
        }

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict


class PlGNNTestonValModule(PlGNNModule):
    """Given a preset evaluation interval, run test dataset when meet the interval.
        model (nn.Module): Model to be trained or evaluated.
        loss_criterion (nn.Module) : Loss compute module.
        evaluator (Metric): Evaluator for evaluating model performance.
        truth_fn (Callable): A function to get true label from batched_data.
        loader (Dict): Arguments loader.
    """

    def __init__(self,
                 model: nn.Module,
                 loss_criterion: nn.Module,
                 evaluator: Metric,
                 truth_fn: Callable,
                 loader
                 ):
        super().__init__(model,
                         loss_criterion,
                         evaluator,
                         truth_fn,
                         loader)
        self.test_eval_still = loader.train.test_eval_interval

    def validation_step(self,
                        batch: Data,
                        batch_idx: Tensor,
                        data_loader_idx: int) -> Dict:

        if data_loader_idx == 0:
            y = self.truth_fn(batch)
            out = self.model(batch)
            loss = self.loss_criterion(out, y)
            self.log("val/loss",
                     loss,
                     prog_bar=False,
                     batch_size=self.loader.train.batch_size,
                     sync_dist=True,
                     add_dataloader_idx=False)
        else:
            if self.test_eval_still != 0:
                return {'loader_idx': data_loader_idx}
            else:
                y = self.truth_fn(batch)
                out = self.model(batch)
                loss = self.loss_criterion(out, y)
                self.log("test/loss",
                         loss,
                         prog_bar=False,
                         batch_size=self.loader.train.batch_size,
                         sync_dist=True,
                         add_dataloader_idx=False)
        return {'loss': loss, 'preds': out, 'target': y, 'loader_idx': data_loader_idx}

    def validation_step_end(self,
                            outputs: Dict) -> None:
        data_loader_idx = outputs["loader_idx"]
        if data_loader_idx == 0:
            self.val_evaluator.update(outputs["preds"], outputs["target"])
        else:
            if self.test_eval_still != 0:
                return
            else:
                self.test_evaluator.update(outputs["preds"], outputs["target"])

    def validation_epoch_end(self,
                             outputs: Dict) -> None:
        self.log("val/metric",
                 self.val_evaluator.compute(),
                 on_epoch=True,
                 on_step=False,
                 prog_bar=True,
                 add_dataloader_idx=False)
        self.val_evaluator.reset()
        if self.test_eval_still == 0:
            self.log("test/metric",
                     self.test_evaluator.compute(),
                     on_epoch=True,
                     on_step=False,
                     prog_bar=True,
                     add_dataloader_idx=False)
            self.test_evaluator.reset()
            self.test_eval_still = self.loader.train.test_eval_interval
        else:
            self.test_eval_still = self.test_eval_still - 1

    def set_test_eval_still(self):
        self.test_eval_still = 0
