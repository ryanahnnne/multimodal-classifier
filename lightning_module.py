"""
Multimodal Binary Classifier - Lightning Module
===============================================
LightningModule wrapping MultimodalClassifier with torchmetrics,
threshold search, and test artifact generation.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import pytorch_lightning as pl
import torchmetrics
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score, average_precision_score

from model.classifier import MultimodalClassifier
from losses import build_criterion, write_test_report


class MultimodalClassifierLightningModule(pl.LightningModule):
    """Lightning wrapper around MultimodalClassifier.

    Handles training/validation/test steps, torchmetrics,
    optimizer/scheduler configuration, threshold search,
    and test artifact generation.
    """

    def __init__(self, model: MultimodalClassifier, cfg: DictConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.train_cfg = cfg.train

        # Loss
        self.criterion = build_criterion(self.train_cfg)

        # torchmetrics for each phase
        for phase in ("train", "val", "test"):
            setattr(self, f"{phase}_accuracy", torchmetrics.Accuracy(task="binary"))
            setattr(self, f"{phase}_precision_metric", torchmetrics.Precision(task="binary"))
            setattr(self, f"{phase}_recall_metric", torchmetrics.Recall(task="binary"))
            setattr(self, f"{phase}_f1", torchmetrics.F1Score(task="binary"))
            setattr(self, f"{phase}_auroc", torchmetrics.AUROC(task="binary"))

        # Accumulators for test-phase threshold search and artifact generation
        self._test_logits: List[torch.Tensor] = []
        self._test_targets: List[torch.Tensor] = []
        self._test_paths: List[str] = []
        self._test_losses: List[float] = []

        # Val logits/targets for threshold search during test phase
        self._val_threshold_logits: List[torch.Tensor] = []
        self._val_threshold_targets: List[torch.Tensor] = []

        self.save_hyperparameters(ignore=["model"])

    # ----------------------------------------------------------
    # Checkpoint: save only trainable parameters
    # ----------------------------------------------------------

    def on_save_checkpoint(self, checkpoint):
        """Save only trainable parameters to reduce checkpoint size."""
        full_state = checkpoint["state_dict"]
        trainable_keys = {
            name for name, param in self.named_parameters()
            if param.requires_grad
        }
        filtered_state = {
            k: v for k, v in full_state.items() if k in trainable_keys
        }

        full_size = sum(v.numel() for v in full_state.values())
        filtered_size = sum(v.numel() for v in filtered_state.values())
        print(f"\nCheckpoint: saving {len(filtered_state)}/{len(full_state)} params "
              f"({filtered_size:,} / {full_size:,} elements, "
              f"{100 * filtered_size / full_size:.1f}%)")

        checkpoint["state_dict"] = filtered_state
        checkpoint["trainable_only"] = True

    def on_load_checkpoint(self, checkpoint):
        """Load trainable-only checkpoint by merging with current model weights."""
        if checkpoint.get("trainable_only", False):
            saved_state = checkpoint["state_dict"]
            full_state = self.state_dict()
            full_state.update(saved_state)
            checkpoint["state_dict"] = full_state

    # ----------------------------------------------------------
    # Text encoding helper
    # ----------------------------------------------------------

    def _encode_texts(self, text_info: List[str]) -> Optional[torch.Tensor]:
        """Encode text strings to features using the model's text encoder."""
        if not self.model.use_text or self.model.text_encoder is None:
            return None
        texts = {"text_info": text_info}
        return self.model.encode_text(texts, self.device)

    # ----------------------------------------------------------
    # Shared step
    # ----------------------------------------------------------

    def _shared_step(self, batch, phase: str):
        """Common forward + loss + metric update logic."""
        images, labels, paths, text_info = batch
        text_features = self._encode_texts(text_info)
        logits = self.model(images, text_features)
        loss = self.criterion(logits, labels)

        # torchmetrics update only (compute/log/reset in epoch_end hooks)
        preds_prob = torch.sigmoid(logits)
        targets_int = labels.int()
        for attr_suffix in ("accuracy", "precision_metric", "recall_metric", "f1", "auroc"):
            getattr(self, f"{phase}_{attr_suffix}").update(preds_prob, targets_int)

        # Log loss
        self.log(
            f"{phase}_loss", loss,
            on_step=(phase == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=images.size(0),
        )

        return loss, logits, labels, paths

    # ----------------------------------------------------------
    # Metrics: compute, log scalar, reset (once per epoch)
    # ----------------------------------------------------------

    def _log_and_reset_metrics(self, phase: str):
        """Compute, log scalar values, and reset all torchmetrics for a phase."""
        metric_map = [
            ("accuracy", "accuracy"),
            ("precision_metric", "precision"),
            ("recall_metric", "recall"),
            ("f1", "f1"),
            ("auroc", "auroc"),
        ]
        for attr_suffix, log_suffix in metric_map:
            metric = getattr(self, f"{phase}_{attr_suffix}")
            self.log(
                f"{phase}_{log_suffix}", metric.compute(),
                prog_bar=(phase == "val"),
            )
            metric.reset()

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._shared_step(batch, "train")
        # Log learning rates
        opt = self.optimizers()
        for i, pg in enumerate(opt.param_groups):
            name = pg.get("name", f"group_{i}")
            self.log(f"lr/{name}", pg["lr"], on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self):
        self._log_and_reset_metrics("train")

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        loss, _, _, _ = self._shared_step(batch, "val")
        return loss

    def on_validation_epoch_end(self):
        self._log_and_reset_metrics("val")

    # ----------------------------------------------------------
    # Test (supports two dataloaders: test=0, val=1 for threshold)
    # ----------------------------------------------------------

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            # Test set evaluation
            loss, logits, labels, paths = self._shared_step(batch, "test")
            self._test_logits.append(logits.detach().cpu())
            self._test_targets.append(labels.detach().cpu())
            self._test_paths.extend(paths)
            self._test_losses.append(loss.detach().cpu().item())
        else:
            # Val set for threshold search (dataloader_idx == 1)
            images, labels, _, text_info = batch
            text_features = self._encode_texts(text_info)
            logits = self.model(images, text_features)
            self._val_threshold_logits.append(logits.detach().cpu())
            self._val_threshold_targets.append(labels.detach().cpu())

    def on_test_epoch_end(self):
        """Compute final metrics, run threshold search, save artifacts."""
        # 1. Gather accumulated tensors
        test_logits = torch.cat(self._test_logits)
        test_targets = torch.cat(self._test_targets)
        test_probs = torch.sigmoid(test_logits)
        avg_loss = sum(self._test_losses) / max(len(self._test_losses), 1)

        # 2. Threshold search on validation logits
        if self._val_threshold_logits:
            val_logits = torch.cat(self._val_threshold_logits)
            val_targets = torch.cat(self._val_threshold_targets)
            opt_threshold_f1, opt_f1_val = self._find_optimal_threshold(val_logits, val_targets, "f1")
            opt_threshold_acc, opt_acc_val = self._find_optimal_threshold(val_logits, val_targets, "accuracy")
        else:
            opt_threshold_f1, opt_f1_val = 0.5, 0.0
            opt_threshold_acc, opt_acc_val = 0.5, 0.0

        # 3. Compute test metrics at default (0.5) and optimal thresholds
        test_at_default = self._compute_at_threshold(test_probs, test_targets, 0.5)
        test_at_opt_f1 = self._compute_at_threshold(test_probs, test_targets, opt_threshold_f1)
        test_at_opt_acc = self._compute_at_threshold(test_probs, test_targets, opt_threshold_acc)

        # 4. AUROC and PR-AUC via sklearn
        targets_np = test_targets.numpy()
        probs_np = test_probs.numpy()
        try:
            auroc = float(roc_auc_score(targets_np, probs_np))
        except ValueError:
            auroc = float("nan")
        try:
            pr_auc = float(average_precision_score(targets_np, probs_np))
        except ValueError:
            pr_auc = float("nan")

        # 5. Build comprehensive test_metrics dict
        test_metrics = {
            "loss": avg_loss,
            **test_at_default,
            "auroc": auroc,
            "pr_auc": pr_auc,
            "optimal_threshold_f1": float(opt_threshold_f1),
            "optimal_threshold_acc": float(opt_threshold_acc),
            "at_optimal_f1_threshold": test_at_opt_f1,
            "at_optimal_acc_threshold": test_at_opt_acc,
        }

        # 6. Print results
        self._print_test_results(
            test_metrics, opt_threshold_f1, opt_f1_val,
            opt_threshold_acc, opt_acc_val, test_at_opt_f1, test_at_opt_acc,
        )

        # 7. Build per-sample predictions
        preds = (test_probs >= 0.5).int()
        predictions = []
        for i, path in enumerate(self._test_paths):
            predictions.append({
                "path": path,
                "gt": int(test_targets[i].item()),
                "pred": int(preds[i].item()),
                "prob": float(test_probs[i].item()),
            })

        # 8. Save artifacts
        output_dir = Path(self.trainer.default_root_dir)
        self._save_artifacts(output_dir, test_metrics, predictions)

        # 9. Cleanup
        self._test_logits.clear()
        self._test_targets.clear()
        self._test_paths.clear()
        self._test_losses.clear()
        self._val_threshold_logits.clear()
        self._val_threshold_targets.clear()

    # ----------------------------------------------------------
    # Optimizer & Scheduler
    # ----------------------------------------------------------

    def configure_optimizers(self):
        """Configure AdamW with param groups + scheduler."""
        lora_lr = self.train_cfg.get("lora_lr", None) or self.train_cfg.learning_rate
        backbone_lr = self.train_cfg.get("backbone_lr", None)
        base_lr = self.train_cfg.learning_rate

        backbone_params = []
        lora_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_" in name:
                lora_params.append(param)
            elif "vision_encoder." in name and "lora_" not in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = []
        if head_params:
            param_groups.append({"params": head_params, "lr": base_lr, "name": "head"})
        if lora_params:
            param_groups.append({"params": lora_params, "lr": lora_lr, "name": "lora"})
        if backbone_params:
            bb_lr = backbone_lr if backbone_lr is not None else base_lr
            param_groups.append({"params": backbone_params, "lr": bb_lr, "name": "backbone"})
        if not param_groups:
            param_groups = [
                {"params": [p for p in self.model.parameters() if p.requires_grad], "lr": base_lr}
            ]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.train_cfg.weight_decay)

        print(f"\nOptimizer param groups:")
        for g in optimizer.param_groups:
            n_params = sum(p.numel() for p in g["params"])
            print(f"  {g.get('name', 'default'):>8s}: lr={g['lr']:.1e}, params={n_params:,}")

        # Scheduler
        scheduler_config = None
        if self.train_cfg.scheduler_type == "cosine_warmup":
            total_steps = self.trainer.estimated_stepping_batches
            max_lrs = [g["lr"] for g in optimizer.param_groups]
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lrs,
                total_steps=total_steps,
                pct_start=self.train_cfg.warmup_ratio,
                anneal_strategy="cos",
            )
            scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        elif self.train_cfg.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.train_cfg.epochs,
            )
            scheduler_config = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

        if scheduler_config:
            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
        return optimizer

    # ----------------------------------------------------------
    # Threshold search helpers
    # ----------------------------------------------------------

    @staticmethod
    def _find_optimal_threshold(logits, targets, metric="f1"):
        """Find optimal threshold by sweeping 0.1-0.9 in steps of 0.01."""
        probs = torch.sigmoid(logits)
        best_threshold, best_value = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.01):
            preds = (probs >= t).float()
            tp = ((preds == 1) & (targets == 1)).sum().float()
            fp = ((preds == 1) & (targets == 0)).sum().float()
            fn = ((preds == 0) & (targets == 1)).sum().float()
            tn = ((preds == 0) & (targets == 0)).sum().float()
            if metric == "f1":
                p = tp / (tp + fp + 1e-8)
                r = tp / (tp + fn + 1e-8)
                value = (2 * p * r / (p + r + 1e-8)).item()
            else:  # accuracy
                value = ((tp + tn) / (tp + tn + fp + fn + 1e-8)).item()
            if value > best_value:
                best_value = value
                best_threshold = t
        return best_threshold, best_value

    @staticmethod
    def _compute_at_threshold(probs, targets, t):
        """Compute binary metrics at a given threshold."""
        preds = (probs >= t).float()
        tp = ((preds == 1) & (targets == 1)).sum().float()
        tn = ((preds == 0) & (targets == 0)).sum().float()
        fp = ((preds == 1) & (targets == 0)).sum().float()
        fn = ((preds == 0) & (targets == 1)).sum().float()
        p = (tp / (tp + fp + 1e-8)).item()
        r = (tp / (tp + fn + 1e-8)).item()
        return {
            "accuracy": ((tp + tn) / (tp + tn + fp + fn + 1e-8)).item(),
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / (p + r + 1e-8),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }

    # ----------------------------------------------------------
    # Output helpers
    # ----------------------------------------------------------

    def _print_test_results(self, metrics, opt_t_f1, opt_f1_val,
                            opt_t_acc, opt_acc_val, at_f1, at_acc):
        """Print test results to stdout."""
        print(f"\nTest Results:")
        print(f"  Loss:        {metrics['loss']:.4f}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  F1 Score:    {metrics['f1']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        auroc = metrics.get("auroc", float("nan"))
        if not math.isnan(auroc):
            print(f"  AUROC:       {auroc:.4f}")
        pr_auc = metrics.get("pr_auc", float("nan"))
        if not math.isnan(pr_auc):
            print(f"  PR-AUC:      {pr_auc:.4f}")
        print(
            f"  Confusion:   TP={metrics['tp']} TN={metrics['tn']} "
            f"FP={metrics['fp']} FN={metrics['fn']}"
        )
        print(f"\n  Optimal Thresholds (from validation set):")
        print(f"    Best F1 threshold:  {opt_t_f1:.2f} (val F1={opt_f1_val:.4f})")
        print(
            f"      -> Test: Acc={at_f1['accuracy']:.4f} F1={at_f1['f1']:.4f} "
            f"P={at_f1['precision']:.4f} R={at_f1['recall']:.4f}"
        )
        print(f"    Best Acc threshold: {opt_t_acc:.2f} (val Acc={opt_acc_val:.4f})")
        print(
            f"      -> Test: Acc={at_acc['accuracy']:.4f} F1={at_acc['f1']:.4f} "
            f"P={at_acc['precision']:.4f} R={at_acc['recall']:.4f}"
        )

    def _save_artifacts(self, output_dir: Path, test_metrics: Dict, predictions: List[Dict]):
        """Save test_predictions.json, results.json, and test_report.md."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        experiment_name = self.train_cfg.experiment_name

        # Determine best epoch from checkpoint callback
        best_epoch = 0
        for cb in self.trainer.callbacks:
            if hasattr(cb, "best_model_score") and hasattr(cb, "best_model_path"):
                # ModelCheckpoint callback
                best_epoch = self.trainer.current_epoch
                break

        # test_predictions.json
        predictions_path = output_dir / "test_predictions.json"
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment": experiment_name,
                    "best_epoch": best_epoch,
                    "test_metrics": test_metrics,
                    "predictions": predictions,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Test predictions saved to {predictions_path}")

        # results.json
        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(
                {
                    "experiment": experiment_name,
                    "best_epoch": best_epoch,
                    "test_metrics": test_metrics,
                },
                f,
                indent=2,
            )
        print(f"Results saved to {results_path}")

        # test_report.md
        write_test_report(
            output_dir=output_dir,
            experiment_name=experiment_name,
            best_epoch=best_epoch,
            test_metrics=test_metrics,
            test_predictions=predictions,
        )
