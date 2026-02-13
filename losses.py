"""
Ad Image Binary Classifier - Loss Functions & Utilities
========================================================
FocalLoss, LabelSmoothingLoss, build_criterion factory,
and test report generation utility.
"""

import math
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


# ============================================================
# Loss Functions
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.55):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = alpha_t * focal_weight * bce
        return loss.mean()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, criterion: nn.Module, smoothing: float = 0.0):
        super().__init__()
        self.criterion = criterion
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.smoothing > 0:
            # hard label -> soft label: 0 -> smoothing/2, 1 -> 1 - smoothing/2
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.criterion(logits, targets)


def build_criterion(train_cfg: DictConfig) -> nn.Module:
    if train_cfg.loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif train_cfg.loss_type == "focal":
        criterion = FocalLoss(gamma=train_cfg.focal.gamma, alpha=train_cfg.focal.alpha)
    else:
        raise ValueError(f"Unknown loss type: {train_cfg.loss_type}")

    smoothing = train_cfg.get("label_smoothing", 0.0)
    if smoothing > 0:
        criterion = LabelSmoothingLoss(criterion, smoothing)

    return criterion


# ============================================================
# Test Report (Markdown)
# ============================================================

def write_test_report(
    output_dir: Path,
    experiment_name: str,
    best_epoch: int,
    test_metrics: Dict[str, float],
    test_predictions: List[Dict],
) -> Path:
    """Write test set report with wrong predictions first."""
    report_path = output_dir / "test_report.md"
    label_str = lambda x: "True" if x == 1 else "False"

    def _row_sort_key(r):
        gt, pred = r["gt"], r["pred"]
        order = {(0, 1): 0, (1, 0): 1, (1, 1): 2, (0, 0): 3}
        return order.get((gt, pred), 4)

    sorted_rows = sorted(test_predictions, key=_row_sort_key)
    n_fp = sum(1 for r in sorted_rows if r["gt"] == 0 and r["pred"] == 1)
    n_fn = sum(1 for r in sorted_rows if r["gt"] == 1 and r["pred"] == 0)
    n_tp = sum(1 for r in sorted_rows if r["gt"] == 1 and r["pred"] == 1)
    n_tn = sum(1 for r in sorted_rows if r["gt"] == 0 and r["pred"] == 0)

    metric_lines = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Loss | {test_metrics['loss']:.4f} |",
        f"| Accuracy | {test_metrics['accuracy']:.4f} |",
        f"| F1 | {test_metrics['f1']:.4f} |",
        f"| Precision | {test_metrics['precision']:.4f} |",
        f"| Recall | {test_metrics['recall']:.4f} |",
    ]
    if "auroc" in test_metrics:
        auroc_val = test_metrics["auroc"]
        metric_lines.append(
            f"| AUROC | {auroc_val:.4f} |" if not math.isnan(auroc_val) else "| AUROC | N/A |"
        )
    if "pr_auc" in test_metrics:
        pr_auc_val = test_metrics["pr_auc"]
        metric_lines.append(
            f"| PR-AUC | {pr_auc_val:.4f} |" if not math.isnan(pr_auc_val) else "| PR-AUC | N/A |"
        )
    metric_lines.append(
        f"| TP / TN / FP / FN | {test_metrics['tp']} / {test_metrics['tn']} / {test_metrics['fp']} / {test_metrics['fn']} |"
    )

    lines = [
        "# Test Set Report",
        "",
        f"**Experiment:** {experiment_name}  ",
        f"**Best Epoch:** {best_epoch}  ",
        "",
        "## Test Metrics Summary",
        "",
        *metric_lines,
        "",
        "## Per-Sample Results (FP / FN first, then TP / TN)",
        "",
        f"FP: {n_fp} | FN: {n_fn} | TP: {n_tp} | TN: {n_tn}",
        "",
        "| # | Image | GT | Pred | Match |",
        "|---|-------|-----|------|-------|",
    ]

    for idx, row in enumerate(sorted_rows, start=1):
        path = row["path"]
        gt = row["gt"]
        pred = row["pred"]
        match = "V" if gt == pred else "X"
        if path:
            file_uri = Path(path).as_uri()
            img_cell = f"[![img]({path})]({file_uri})"
        else:
            img_cell = "-"
        lines.append(f"| {idx} | {img_cell} | {label_str(gt)} | {label_str(pred)} | {match} |")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Test report saved to {report_path}")
    return report_path
