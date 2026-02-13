"""
Experiment result comparison script.
Collects results.json from experiment outputs and generates a comparison table.

Usage:
    # Compare all experiments
    python scripts/compare_results.py

    # Compare specific experiments by pattern
    python scripts/compare_results.py --pattern "directFT_lightning_*"

    # Sort by specific metric (default: test_auroc)
    python scripts/compare_results.py --sort test_f1

    # Export to CSV
    python scripts/compare_results.py --csv results.csv

    # Specify output directory
    python scripts/compare_results.py --output-dir ./outputs
"""

import argparse
import glob
import json
from pathlib import Path


METRICS_KEYS = ["accuracy", "precision", "recall", "f1", "auroc", "pr_auc"]


def find_results(output_dir: str, pattern: str) -> list[dict]:
    """Find all results.json matching the experiment name pattern."""
    results = []
    for results_path in sorted(Path(output_dir).rglob("results.json")):
        try:
            with open(results_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

        exp_name = data.get("experiment", "")
        if not glob.fnmatch.fnmatch(exp_name, pattern):
            continue

        test_metrics = data.get("test_metrics", {})
        entry = {
            "path": str(results_path.parent),
            "experiment": exp_name,
            "best_epoch": data.get("best_epoch", "N/A"),
            **{f"test_{k}": test_metrics.get(k, 0) for k in METRICS_KEYS},
            "test_loss": test_metrics.get("loss", 0),
            "opt_threshold_f1": test_metrics.get("optimal_threshold_f1", 0.5),
            "opt_threshold_acc": test_metrics.get("optimal_threshold_acc", 0.5),
        }

        # Metrics at optimal F1 threshold
        at_opt_f1 = test_metrics.get("at_optimal_f1_threshold", {})
        if at_opt_f1:
            entry["opt_f1_accuracy"] = at_opt_f1.get("accuracy", 0)
            entry["opt_f1_f1"] = at_opt_f1.get("f1", 0)
            entry["opt_f1_precision"] = at_opt_f1.get("precision", 0)
            entry["opt_f1_recall"] = at_opt_f1.get("recall", 0)

        results.append(entry)

    return results


def print_table(results: list[dict], sort_by: str):
    """Print results as a formatted table."""
    if not results:
        print("No results found.")
        return

    # Sort
    results.sort(key=lambda r: r.get(sort_by, 0), reverse=True)

    # Table columns
    columns = [
        ("Experiment", "experiment", 45, "s"),
        ("Epoch", "best_epoch", 5, "d"),
        ("Test AUROC", "test_auroc", 10, ".4f"),
        ("Test F1", "test_f1", 8, ".4f"),
        ("Test Acc", "test_accuracy", 8, ".4f"),
        ("Test P", "test_precision", 7, ".4f"),
        ("Test R", "test_recall", 7, ".4f"),
        ("PR-AUC", "test_pr_auc", 7, ".4f"),
        ("OptT", "opt_threshold_f1", 5, ".2f"),
        ("Opt F1", "opt_f1_f1", 7, ".4f"),
    ]

    # Header
    header = " | ".join(f"{name:>{width}}" for name, _, width, _ in columns)
    sep = "-+-".join("-" * width for _, _, width, _ in columns)
    print(f"\n{header}")
    print(sep)

    # Rows
    best_auroc = max(r.get("test_auroc", 0) for r in results)
    for r in results:
        parts = []
        for _, key, width, fmt in columns:
            val = r.get(key, 0)
            parts.append(f"{val:>{width}{fmt}}")
        line = " | ".join(parts)
        # Mark best
        if r.get("test_auroc", 0) == best_auroc:
            line += "  << BEST"
        print(line)

    print(f"\nTotal: {len(results)} experiments (sorted by {sort_by} desc)")


def export_csv(results: list[dict], csv_path: str, sort_by: str):
    """Export results to CSV."""
    if not results:
        return

    results.sort(key=lambda r: r.get(sort_by, 0), reverse=True)

    keys = [
        "experiment", "best_epoch",
        "test_auroc", "test_f1", "test_accuracy", "test_precision", "test_recall",
        "test_pr_auc", "test_loss",
        "opt_threshold_f1", "opt_f1_f1", "opt_f1_accuracy", "opt_f1_precision", "opt_f1_recall",
        "opt_threshold_acc",
        "path",
    ]

    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in results:
            row = [str(r.get(k, "")) for k in keys]
            f.write(",".join(row) + "\n")

    print(f"\nCSV exported to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--pattern", type=str, default="*",
                        help="Experiment name glob pattern (e.g. 'directFT_lightning_*')")
    parser.add_argument("--output-dir", type=str, default="whole_finetune",
                        help="Output directory (default: whole_finetune)")
    parser.add_argument("--sort", type=str, default="test_auroc",
                        help="Metric to sort by (default: test_auroc)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Export to CSV file path")
    args = parser.parse_args()

    print(f"Searching for experiments matching: {args.pattern}")
    results = find_results(args.output_dir, args.pattern)

    print_table(results, args.sort)

    if args.csv:
        export_csv(results, args.csv, args.sort)


if __name__ == "__main__":
    main()
