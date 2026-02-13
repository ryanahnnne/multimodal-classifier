"""
Usage:
    # Default training (SigLIP2 + LaBSE, attention pooling, gated fusion)
    python train.py

    # Change visual encoder
    python train.py vision_encoder=resnet50
    python train.py vision_encoder=vit
    python train.py vision_encoder=siglip2_base
    python train.py vision_encoder=efficientnet

    # Change text encoder or disable it
    python train.py text_encoder=labse
    python train.py text_encoder=bge_m3
    python train.py text_encoder=e5_large
    python train.py text_encoder=none

    # Change pooling / fusion strategy
    python train.py pooling=mean
    python train.py pooling=attention
    python train.py fusion=concat
    python train.py fusion=gated

    # Change augmentation
    python train.py augmentation=weak
    python train.py augmentation=strong

    # Use experiment preset
    python train.py +experiment=directFT_lightning_1_allLayers_mean
    python train.py +experiment=directFT_lightning_2_allLayers_attn

    # Override training parameters
    python train.py train.batch_size=32 train.epochs=50 train.learning_rate=1e-4

    # Hydra multirun for hyperparameter sweep
    python train.py -m vision_encoder=siglip2_so400m,resnet50 pooling=mean,attention

    # View resolved config without running
    python train.py --cfg job
"""

import logging
import os
import random
import warnings

# Suppress noisy logs before importing other libraries
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*Found .* module.*in eval mode at the start of training.*")

import numpy as np
import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from model import build_classifier
from datamodule import AdImageDataModule
from lightning_module import MultimodalClassifierLightningModule


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    pl.seed_everything(seed, workers=True)


def print_config(cfg: DictConfig):
    """Pretty-print the resolved configuration."""
    print("=" * 60)
    print("Configuration")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("=" * 60)


def build_logger(cfg: DictConfig, output_dir: str):
    """Build Lightning logger from config."""
    logger_cfg = cfg.train.get("logger", {})
    logger_type = logger_cfg.get("type", "wandb")

    if logger_type == "wandb":
        return WandbLogger(
            project=logger_cfg.get("project", "multimodal-classification-model"),
            name=cfg.train.experiment_name,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    elif logger_type == "tensorboard":
        return TensorBoardLogger(
            save_dir=output_dir,
            name="tensorboard",
        )
    else:
        return None


def build_callbacks(cfg: DictConfig, output_dir: str):
    """Build Lightning callbacks."""
    callbacks = []

    save_ckpt = cfg.train.get("save_ckpt", False)

    # ModelCheckpoint - save best model by val_auroc
    ckpt_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="checkpoint_best",
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        save_last=save_ckpt,
        verbose=True,
    )
    callbacks.append(ckpt_callback)

    # EarlyStopping
    es_metric = cfg.train.early_stopping.metric
    es_mode = "min" if "loss" in es_metric else "max"
    callbacks.append(EarlyStopping(
        monitor=es_metric,
        mode=es_mode,
        patience=cfg.train.early_stopping.patience,
        verbose=True,
    ))

    return callbacks


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Args:
        cfg: Hydra-composed configuration
    """
    # Get Hydra output directory (works for both single run and multirun)
    hydra_output_dir = HydraConfig.get().runtime.output_dir

    # Print config
    print_config(cfg)

    # Set seed
    set_seed(cfg.train.seed)

    # Build model (nn.Module, unchanged)
    model = build_classifier(cfg)

    # Lightning wrappers
    lit_module = MultimodalClassifierLightningModule(model, cfg)
    datamodule = AdImageDataModule(cfg)

    # Logger
    logger = build_logger(cfg, hydra_output_dir)

    # Callbacks
    callbacks = build_callbacks(cfg, hydra_output_dir)

    # Precision
    precision = "16-mixed" if (cfg.train.mixed_precision and cfg.train.device == "cuda") else 32

    # Accelerator + devices + strategy
    use_data_parallel = cfg.train.get("use_data_parallel", False)
    if cfg.train.device == "cuda":
        accelerator = "gpu"
        devices = "auto"
        strategy = "ddp" if use_data_parallel else "auto"
    elif cfg.train.device == "mps":
        accelerator = "mps"
        devices = 1
        strategy = "auto"
    else:
        accelerator = "cpu"
        devices = 1
        strategy = "auto"

    # Logging interval
    logger_cfg = cfg.train.get("logger", {})
    log_interval = logger_cfg.get("log_interval", 0)
    log_every_n_steps = max(1, log_interval) if log_interval > 0 else 50

    # Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=hydra_output_dir,
        deterministic=True,
        log_every_n_steps=log_every_n_steps,
    )

    # Train
    trainer.fit(lit_module, datamodule=datamodule)

    # Test with best checkpoint (pass both test + val dataloaders for threshold search)
    ckpt_callback = callbacks[0]  # ModelCheckpoint is first
    best_ckpt_path = ckpt_callback.best_model_path

    # Ensure dataloaders are available for test phase
    datamodule.setup(stage="test")
    trainer.test(
        lit_module,
        dataloaders=[datamodule.test_dataloader(), datamodule.val_dataloader()],
        ckpt_path=best_ckpt_path,
        weights_only=False,
    )

    # Cleanup checkpoints if save_ckpt is disabled
    if not cfg.train.get("save_ckpt", False):
        import glob
        for ckpt_file in glob.glob(os.path.join(hydra_output_dir, "*.ckpt")):
            os.remove(ckpt_file)
            print(f"Removed checkpoint: {ckpt_file}")


if __name__ == "__main__":
    main()
