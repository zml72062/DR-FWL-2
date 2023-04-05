from typing import Tuple, Callable
import pytorch_lightning as pl
from pygmmpp.data import Dataset, DataLoader
from data_utils.batch import collate


class PlPyGDataModule(pl.LightningDataModule):
    """Pytorch lightning data module for PyG dataset.
    Args:
        train_dataset (Dataset): Train PyG dataset.
        val_dataset (Dataset): Validation PyG dataset.
        test_dataset (Dataset): Test PyG dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of process for data loader.
    """

    def __init__(self,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 test_dataset: Dataset,
                 batch_size: int = 32,
                 num_workers: int = 0):
        super(PlPyGDataModule, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          collator=collate)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          collator=collate)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          collator=collate)


class PlPyGDataTestonValModule(PlPyGDataModule):
    def val_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        return (DataLoader(self.val_dataset,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           shuffle=False,
                           collator=collate),
                DataLoader(self.test_dataset,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           shuffle=False,
                           collator=collate))
