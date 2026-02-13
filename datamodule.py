"""
Ad Image Binary Classifier - Lightning DataModule
===================================================
Wraps the existing dataset.py functions into a LightningDataModule.
"""

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

import pytorch_lightning as pl

from dataset import (
    AdImageDataset,
    load_samples_from_csv,
    get_train_transform,
    get_eval_transform,
)


class AdImageDataModule(pl.LightningDataModule):
    """Lightning DataModule for the ad image classification task.

    Wraps existing dataset.py functions without modifying them.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """Load CSV samples and create datasets."""
        if stage == "fit" or stage is None:
            train_samples = load_samples_from_csv(self.cfg.data.train_csv)
            val_samples = load_samples_from_csv(self.cfg.data.val_csv)
            train_transform = get_train_transform(self.cfg)
            eval_transform = get_eval_transform(self.cfg)
            self.train_dataset = AdImageDataset(train_samples, train_transform)
            self.val_dataset = AdImageDataset(val_samples, eval_transform)

        if stage == "test" or stage is None:
            test_samples = load_samples_from_csv(self.cfg.data.test_csv)
            eval_transform = get_eval_transform(self.cfg)
            self.test_dataset = AdImageDataset(test_samples, eval_transform)

        # Also load val dataset for threshold search during test phase
        if stage == "test" and self.val_dataset is None:
            val_samples = load_samples_from_csv(self.cfg.data.val_csv)
            eval_transform = get_eval_transform(self.cfg)
            self.val_dataset = AdImageDataset(val_samples, eval_transform)

    @staticmethod
    def collate_fn(batch):
        """Custom collate handling text strings alongside tensors."""
        images = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        paths = [item[2] for item in batch]
        text_info = [item[3] for item in batch]
        return images, labels, paths, text_info

    def _shared_dataloader_kwargs(self):
        seed = self.cfg.train.get("seed", 42)

        def worker_init_fn(worker_id):
            import random
            import numpy as np
            worker_seed = seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)

        return dict(
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            worker_init_fn=worker_init_fn,
        )

    def train_dataloader(self):
        seed = self.cfg.train.get("seed", 42)
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            generator=g,
            **self._shared_dataloader_kwargs(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            **self._shared_dataloader_kwargs(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            **self._shared_dataloader_kwargs(),
        )
