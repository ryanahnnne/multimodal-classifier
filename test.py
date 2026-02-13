"""
Ad Image Binary Classifier - Test Evaluation
=============================================
Load a trained model and run test evaluation using PyTorch Lightning.
Supports both Lightning .ckpt and legacy .pt checkpoint formats.

Usage:
    # Basic usage with experiment directory
    python test.py --experiment outputs/default/2024-01-01_12-00-00

    # With custom test CSV
    python test.py --experiment outputs/default/... --test_csv /path/to/test.csv

    # Override batch size
    python test.py --experiment outputs/default/... --batch_size 32

    # Use legacy .pt checkpoint
    python test.py --experiment outputs/default/... --checkpoint checkpoint_best.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig

from model import build_classifier
from datamodule import AdImageDataModule
from lightning_module import AdImageLightningModule


def load_config_from_experiment(experiment_dir: Path) -> DictConfig:
    """Load config from experiment directory.

    Tries Hydra config first (.hydra/config.yaml), then falls back to
    searching for config.yaml in common locations.
    """
    # Try Hydra config first
    hydra_config = experiment_dir / ".hydra" / "config.yaml"
    if hydra_config.exists():
        print(f"Loading Hydra config: {hydra_config}")
        return OmegaConf.load(hydra_config)

    # Fallback to common config locations
    possible_configs = [
        experiment_dir / "config.yaml",
        experiment_dir.parent.parent / "conf" / "config.yaml",
        experiment_dir.parent.parent / "cfg" / "config.yaml",
        Path("conf/config.yaml"),
        Path("cfg/config.yaml"),
    ]

    for config_path in possible_configs:
        if config_path.exists():
            print(f"Loading config: {config_path}")
            return OmegaConf.load(config_path)

    raise FileNotFoundError(
        f"Could not find config.yaml. Searched:\n"
        + "\n".join(f"  - {p}" for p in [hydra_config] + possible_configs)
    )


def run_test(
    experiment_dir: str,
    test_csv: str = None,
    batch_size: int = None,
    checkpoint_name: str = "checkpoint_best.ckpt",
    output_suffix: str = None,
):
    """Run test evaluation on a trained model."""
    experiment_dir = Path(experiment_dir)

    if not experiment_dir.exists():
        print(f"ERROR: Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    checkpoint_path = experiment_dir / checkpoint_name
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load config
    cfg = load_config_from_experiment(experiment_dir)

    # Override test CSV if provided
    if test_csv:
        cfg.data.test_csv = test_csv

    # Override batch size if provided
    if batch_size:
        cfg.train.batch_size = batch_size

    # Build model and Lightning module
    model = build_classifier(cfg)
    lit_module = AdImageLightningModule(model, cfg)

    # Handle legacy .pt checkpoints (manual state_dict loading)
    if checkpoint_name.endswith(".pt"):
        print(f"Loading legacy .pt checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        lit_module.model.load_state_dict(checkpoint["model_state_dict"])
        if "epoch" in checkpoint:
            print(f"  Checkpoint epoch: {checkpoint['epoch']}")
        if "metrics" in checkpoint:
            metrics = checkpoint["metrics"]
            auroc = metrics.get("auroc", "N/A")
            f1 = metrics.get("f1", "N/A")
            if isinstance(auroc, float):
                auroc = f"{auroc:.4f}"
            if isinstance(f1, float):
                f1 = f"{f1:.4f}"
            print(f"  Checkpoint metrics: AUROC={auroc}, F1={f1}")
        ckpt_path = None  # Don't pass ckpt_path to trainer.test when manually loaded
    else:
        ckpt_path = str(checkpoint_path)

    # DataModule
    datamodule = AdImageDataModule(cfg)
    datamodule.setup(stage="test")

    # Override output dir if suffix provided
    if output_suffix:
        output_dir = str(experiment_dir)
        cfg.train.experiment_name = f"{cfg.train.experiment_name}_{output_suffix}"
    else:
        output_dir = str(experiment_dir)

    # Determine precision + accelerator
    if cfg.train.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        accelerator = "cpu"
        precision = 32
    elif cfg.train.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        accelerator = "cpu"
        precision = 32
    elif cfg.train.device == "cuda":
        accelerator = "gpu"
        precision = "16-mixed" if cfg.train.mixed_precision else 32
    elif cfg.train.device == "mps":
        accelerator = "mps"
        precision = 32
    else:
        accelerator = "cpu"
        precision = 32

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        precision=precision,
        default_root_dir=output_dir,
    )

    trainer.test(
        lit_module,
        dataloaders=[datamodule.test_dataloader(), datamodule.val_dataloader()],
        ckpt_path=ckpt_path,
        weights_only=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test evaluation for trained Ad Image Classifier",
        epilog=(
            "Examples:\n"
            "  python test.py --experiment outputs/default/2024-01-01_12-00-00\n"
            "  python test.py --experiment outputs/step5/... --test_csv /path/to/new_test.csv\n"
            "  python test.py --experiment outputs/default/... --batch_size 32\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Path to experiment output directory",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="Optional: Override test CSV path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional: Override batch size for evaluation",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint_best.ckpt",
        help="Checkpoint filename (default: checkpoint_best.ckpt)",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default=None,
        help="Optional suffix for output files (e.g., 'new_testset')",
    )

    args = parser.parse_args()

    run_test(
        experiment_dir=args.experiment,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        checkpoint_name=args.checkpoint,
        output_suffix=args.output_suffix,
    )


if __name__ == "__main__":
    main()
