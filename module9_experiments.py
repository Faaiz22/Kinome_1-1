"""
module9_experiments.py
======================
TITLE
Experiment orchestration and model comparison layer.

PURPOSE
This module runs larger comparison studies across model families and seeds while
tracking time budgets, uncertainty outputs, and ranked results.

WHAT IT DOES
- Trains or reuses multiple model configs.
- Runs post-training evaluation and uncertainty analysis.
- Aggregates metrics across seeds and model families.
- Writes experiment summaries and statistical comparison artefacts.

HOW IT WORKS
1. Configure requested configs, seeds, and budget.
2. Train or reuse checkpoints.
3. Evaluate each run and collect metrics.
4. Export ranked results, reliability data, and summary text.

INPUT CONTRACT
- Clean aligned dataset and built feature stores.
- Valid config IDs and checkpoint/output directories.

OUTPUT CONTRACT
- `results.csv`, `per_seed_results.csv`, `per_kinase_results.csv`,
  uncertainty/calibration files, and experiment summaries.

DEPENDENCIES
- module6_training.py
- module7_uncertainty.py
- module8_evaluation.py

CRITICAL ASSUMPTIONS
- Requested configs are compatible with the selected family/task setup.
- The time budget is a hard operational constraint.

FAILURE MODES
- Missing checkpoints or feature stores
- Budget exhaustion
- Failed individual config runs

SAFETY CHECKS IMPLEMENTED
- Partial-result flushing
- Time-bounded execution
- Checkpoint reuse and per-run error capture

HOW TO RUN
- `python module9_experiments.py --config all --seeds 42 52 62`

HOW IT CONNECTS TO PIPELINE
It sits above module6 and module8 to compare multiple trained configurations and
feed downstream reporting and decision-making.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from progress_utils import progress_iter

from module2_feature_engineering import load_ligand_feature_store
from module3_protein_features import load_protein_feature_store
from module5_models import ALL_CONFIG_IDS, get_model_config
from module6_training import (
    KinaseLigandDataset,
    TrainConfig,
    collate_fn,
    load_checkpoint,
    load_saved_split_indices,
    murcko_scaffold_split,
    predict,
    set_seed,
    train_config,
    train_all_configs,
    DEVICE,
)
from module7_uncertainty import (
    UncertaintyEnsemble,
    UncertaintyResult,
    compute_calibration_metrics,
    load_ensemble,
    reliability_diagram_data,
)
from module8_evaluation import (
    aggregate_seed_metrics,
    build_results_csv,
    classification_metrics,
    evaluate_predictions,
    run_full_evaluation,
    save_per_kinase_results,
    spearman_rho,
    rmse,
    compute_ef1_percent,
    calibration_spearman,
)

log = logging.getLogger("module9")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Experiment configuration
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@dataclass
class ExperimentConfig:
    """
    Top-level experiment configuration.

    Attributes
    ----------
    config_ids      : which model configs to run ('all' â†’ ALL_CONFIG_IDS)
    seeds           : random seeds for multi-seed training
    train_cfg       : TrainConfig passed to module6
    checkpoint_dir  : directory for model checkpoints
    results_dir     : directory for output files
    threshold       : pIC50 threshold for EF1% active/inactive split
    device          : training/inference device
    skip_training   : if True, skip training and load existing checkpoints
    run_pairwise    : run Wilcoxon pairwise significance tests
    fast_debug      : use tiny dataset / 2 epochs for quick debugging
    """
    config_ids:     list[str]  = field(default_factory=lambda: ALL_CONFIG_IDS)
    seeds:          list[int]  = field(default_factory=lambda: [42])
    train_cfg:      TrainConfig = field(default_factory=TrainConfig)
    checkpoint_dir: str        = "./checkpoints"
    results_dir:    str        = "./results"
    threshold:      float      = 7.0
    device:         str        = DEVICE
    skip_training:  bool       = False
    run_pairwise:   bool       = True
    fast_debug:     bool       = False


PRIORITY_CONFIGS: list[str] = [
    "full_model",
    "cross_attention",
    "ligand_plus_protein",
    "ligand_only",
    "protein_only",
]


def _format_hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _remaining_seconds(start_time: float, limit_hours: float = 24.0) -> float:
    return max(0.0, limit_hours * 3600 - (time.time() - start_time))


def _family_checkpoint_dir(root: str, family: str) -> str:
    safe = family.replace("-", "_")
    return str(Path(root) / safe)


def _make_family_train_cfg(
    base_cfg: TrainConfig,
    family: str,
    checkpoint_root: str,
    seeds: list[int],
    start_time: float,
    limit_hours: float = 24.0,
) -> TrainConfig:
    cfg = copy.deepcopy(base_cfg)
    cfg.seeds = seeds
    cfg.family = family
    cfg.global_start_time = start_time
    cfg.global_time_limit_hours = limit_hours
    cfg.checkpoint_dir = _family_checkpoint_dir(checkpoint_root, family)
    if family == "regression":
        cfg.task_type_override = "regression"
        cfg.label_scheme = "pIC50_continuous"
    elif family == "hard_classification":
        cfg.task_type_override = "classification"
        cfg.label_scheme = "IC50_500_5000_hard"
        cfg.use_sample_weight = False
    else:
        raise ValueError(f"Unknown family '{family}'")
    return cfg


def _family_dataset_stats(dataset_df: pd.DataFrame, family: str, train_cfg: TrainConfig) -> dict:
    if family == "hard_classification":
        mask = (
            (dataset_df["ic50_nm_median"] < train_cfg.active_cutoff_nm) |
            (dataset_df["ic50_nm_median"] > train_cfg.inactive_cutoff_nm)
        )
        view = dataset_df[mask].copy()
        labels = (view["ic50_nm_median"] < train_cfg.active_cutoff_nm).astype(int)
        dropped = int((~mask).sum())
    elif family == "posthoc":
        view = dataset_df.copy()
        labels = (view["pIC50"] >= train_cfg.binary_threshold_pic50).astype(int)
        dropped = 0
    else:
        view = dataset_df.copy()
        labels = (view["pIC50"] >= train_cfg.binary_threshold_pic50).astype(int)
        dropped = 0

    return {
        "n_samples": int(len(view)),
        "dropped_rows": dropped,
        "n_active": int(labels.sum()),
        "n_inactive": int(len(labels) - labels.sum()),
        "n_unique_ligands": int(view["inchikey"].nunique()),
        "n_unique_targets": int(view["uniprot_id"].nunique()),
    }


def _estimate_next_run_seconds(history: list[float], fallback_hours: float = 1.5) -> float:
    if history:
        return float(np.mean(history))
    return fallback_hours * 3600


def _flush_partial_results(results_dir: Path, run_rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed_df = pd.DataFrame(run_rows)
    per_seed_df.to_csv(results_dir / "per_seed_results.csv", index=False)

    if per_seed_df.empty:
        results_df = pd.DataFrame()
    else:
        group_cols = ["family", "config_id", "task_type", "label_scheme"]
        rows = []
        numeric_cols = [
            c for c in per_seed_df.columns
            if c not in group_cols + ["seed", "checkpoint_path", "status", "error", "split_sizes"]
            and pd.api.types.is_numeric_dtype(per_seed_df[c])
        ]
        for keys, group in per_seed_df.groupby(group_cols, dropna=False):
            row = dict(zip(group_cols, keys))
            row["n_completed_seeds"] = int(group["seed"].nunique())
            statuses = sorted(set(group["status"].dropna().tolist()))
            row["status"] = "completed" if statuses == ["completed"] else ",".join(statuses)
            for col in numeric_cols:
                vals = group[col].dropna().astype(float)
                if len(vals) == 0:
                    continue
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_std"] = float(vals.std(ddof=0))
            rows.append(row)
        results_df = pd.DataFrame(rows)
        results_df.to_csv(results_dir / "results.csv", index=False)

    summary_lines = [
        "Time-bounded ablation study summary",
        f"Runs completed: {len(run_rows)}",
    ]
    if not results_df.empty:
        for family, fam_df in results_df.groupby("family"):
            summary_lines.append(f"\n[{family}]")
            metric = "spearman_mean" if "spearman_mean" in fam_df.columns else "roc_auc_mean"
            fam_sorted = fam_df.sort_values(metric, ascending=False) if metric in fam_df.columns else fam_df
            for _, row in fam_sorted.iterrows():
                summary_lines.append(
                    f"{row['config_id']}: {metric}={row.get(metric, float('nan')):.4f} status={row.get('status','')}"
                )
    (results_dir / "experiment_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    return per_seed_df, results_df


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Per-seed evaluation (single config, single seed)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def evaluate_single_seed(
    config_id:     str,
    seed:          int,
    dataset:       KinaseLigandDataset,
    checkpoint_dir: str,
    threshold:     float,
    device:        str,
) -> dict:
    """
    Load the checkpoint for (config_id, seed), run inference on the test split,
    and return a flat metrics dict for this seed.

    Returns
    -------
    dict with scalar metrics for this (config_id, seed).
    """
    from torch.utils.data import DataLoader, Subset

    model, ckpt = load_checkpoint(config_id, seed, checkpoint_dir, device)
    model_cfg = model.cfg

    test_idx = load_saved_split_indices(config_id, seed, checkpoint_dir)

    test_ds = Subset(dataset, test_idx)
    test_loader = DataLoader(
        test_ds, batch_size=128, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    preds = predict(model, test_loader, device)
    if model_cfg.task_type == "classification":
        cls = classification_metrics(preds["labels"].astype(int), preds["probs"])
        return {
            "config_id":        config_id,
            "task_type":        model_cfg.task_type,
            "seed":             seed,
            "roc_auc":          cls["roc_auc"],
            "pr_auc":           cls["pr_auc"],
            "accuracy":         cls["accuracy"],
            "f1":               cls["f1"],
            "n_test":           len(preds["labels"]),
            "val_loss":         ckpt.get("val_loss", float("nan")),
        }

    y_true = preds["targets"]
    y_pred = preds["mu"]
    sigma  = np.sqrt(np.clip(preds["var"], 1e-8, None))

    rho   = spearman_rho(y_true, y_pred)
    rmse_ = rmse(y_true, y_pred)
    ef1, ef1_valid = compute_ef1_percent(y_true, y_pred, threshold)
    cal   = calibration_spearman(y_true, y_pred, sigma)
    cls = classification_metrics((y_true >= threshold).astype(int), y_pred)

    return {
        "config_id":        config_id,
        "task_type":        model_cfg.task_type,
        "seed":             seed,
        "spearman":         rho,
        "rmse":             rmse_,
        "ef1pct":           ef1,
        "ef1pct_is_valid":  ef1_valid,
        "calibration":      cal,
        "roc_auc":          cls["roc_auc"],
        "pr_auc":           cls["pr_auc"],
        "accuracy":         cls["accuracy"],
        "f1":               cls["f1"],
        "n_test":           len(y_true),
        "val_loss":         ckpt.get("val_loss", float("nan")),
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Multi-seed aggregation for a single config
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def aggregate_config_seeds(
    config_id: str,
    seed_results: list[dict],
) -> dict:
    """
    Aggregate per-seed metrics to mean Â± std.

    Decision rule applied:
        meaningful_gain[metric] = True iff Î” > std
    (Î” is computed relative to the ligand_only baseline in run_all_experiments.)

    Returns a flat dict with all aggregate metrics.
    """
    task_type = seed_results[0].get("task_type", get_model_config(config_id).task_type) if seed_results else get_model_config(config_id).task_type
    metrics_to_agg = ["roc_auc", "pr_auc", "accuracy", "f1"]
    if task_type == "regression":
        metrics_to_agg = ["spearman", "rmse", "ef1pct", "calibration", "roc_auc", "pr_auc", "accuracy", "f1"]
    agg = {"config_id": config_id, "task_type": task_type}

    for m in metrics_to_agg:
        vals = [r[m] for r in seed_results
                if isinstance(r.get(m), float) and not np.isnan(r.get(m, np.nan))]
        if vals:
            agg[f"{m}_mean"] = float(np.mean(vals))
            agg[f"{m}_std"]  = float(np.std(vals, ddof=0))
            agg[f"{m}_min"]  = float(np.min(vals))
            agg[f"{m}_max"]  = float(np.max(vals))
            agg[f"{m}_values"] = vals          # kept for Wilcoxon test
        else:
            agg[f"{m}_mean"] = float("nan")
            agg[f"{m}_std"]  = float("nan")

    agg["n_seeds"]        = len(seed_results)
    agg["n_test_samples"] = seed_results[0].get("n_test", 0) if seed_results else 0
    agg["ef1pct_is_valid"] = any(r.get("ef1pct_is_valid", False) for r in seed_results)

    return agg


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Pairwise Wilcoxon significance test
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def pairwise_wilcoxon(
    config_agg_results: dict,   # {config_id: agg_dict}
    metric: str = "spearman",
    alpha:  float = 0.05,
) -> pd.DataFrame:
    """
    Compute Wilcoxon signed-rank test p-values for all pairs of configurations.

    Uses per-seed values (not aggregated means) to maximise statistical power.
    With only 3 seeds, results should be interpreted cautiously.

    Parameters
    ----------
    config_agg_results : {config_id: agg_dict}  (must contain "{metric}_values")
    metric             : which metric to compare
    alpha              : significance level

    Returns
    -------
    pd.DataFrame with columns: config_a, config_b, p_value, significant, delta_mean
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        log.warning("scipy.stats.wilcoxon not available â€” skipping pairwise tests.")
        return pd.DataFrame()

    config_ids = list(config_agg_results.keys())
    rows = []

    for i in range(len(config_ids)):
        for j in range(i + 1, len(config_ids)):
            cid_a = config_ids[i]
            cid_b = config_ids[j]

            vals_a = config_agg_results[cid_a].get(f"{metric}_values", [])
            vals_b = config_agg_results[cid_b].get(f"{metric}_values", [])

            if len(vals_a) < 2 or len(vals_b) < 2 or len(vals_a) != len(vals_b):
                p_val = float("nan")
                sig   = False
            else:
                try:
                    # Wilcoxon requires non-zero differences
                    diffs = [a - b for a, b in zip(vals_a, vals_b)]
                    if all(d == 0 for d in diffs):
                        p_val = 1.0
                        sig   = False
                    else:
                        _, p_val = wilcoxon(vals_a, vals_b, alternative="two-sided")
                        sig = p_val < alpha
                except Exception:
                    p_val = float("nan")
                    sig   = False

            delta = (
                config_agg_results[cid_a].get(f"{metric}_mean", np.nan) -
                config_agg_results[cid_b].get(f"{metric}_mean", np.nan)
            )

            rows.append({
                "config_a":   cid_a,
                "config_b":   cid_b,
                "metric":     metric,
                "p_value":    p_val,
                "significant": sig,
                "delta_mean": delta,
                "better":     cid_a if delta > 0 else cid_b,
            })

    df = pd.DataFrame(rows)
    log.info("Pairwise Wilcoxon tests (%s): %d pairs, %d significant at Î±=%.2f",
             metric, len(df), df["significant"].sum(), alpha)
    return df


def evaluate_posthoc_classification(
    config_id: str,
    seed: int,
    dataset: KinaseLigandDataset,
    checkpoint_dir: str,
    threshold: float,
    device: str,
) -> dict:
    from torch.utils.data import DataLoader, Subset

    model, ckpt = load_checkpoint(config_id, seed, checkpoint_dir, device)
    test_idx = load_saved_split_indices(config_id, seed, checkpoint_dir)
    test_ds = Subset(dataset, test_idx)
    test_loader = DataLoader(
        test_ds, batch_size=128, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    preds = predict(model, test_loader, device)
    cls = classification_metrics((preds["targets"] >= threshold).astype(int), preds["mu"])
    return {
        "config_id": config_id,
        "task_type": "posthoc",
        "seed": seed,
        "roc_auc": cls["roc_auc"],
        "pr_auc": cls["pr_auc"],
        "accuracy": cls["accuracy"],
        "f1": cls["f1"],
        "n_test": len(preds["targets"]),
        "val_loss": ckpt.get("val_loss", float("nan")),
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Decision rule: meaningful gain filter
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def apply_decision_rule(
    results_df:   pd.DataFrame,
    baseline_id:  str   = "ligand_only",
    metric:       str   = "spearman_mean",
    std_col:      str   = "spearman_std",
) -> pd.DataFrame:
    """
    Apply the decision rule from the spec:
        Reject an apparent improvement if Î”metric < std

    Adds columns:
        delta_vs_baseline : float
        meaningful_gain   : bool (True iff delta > std)

    Parameters
    ----------
    results_df   : results DataFrame (one row per config_id)
    baseline_id  : config used as the reference baseline
    metric       : column to compare
    std_col      : column containing the standard deviation of metric

    Returns
    -------
    Modified DataFrame with decision columns added.
    """
    df = results_df.copy()

    if metric not in df.columns:
        log.warning("Metric '%s' not in results DataFrame â€” skipping decision rule.", metric)
        return df

    baseline_rows = df[df["config_id"] == baseline_id]
    if baseline_rows.empty:
        log.warning("Baseline config '%s' not found â€” skipping decision rule.", baseline_id)
        return df

    baseline_val = baseline_rows[metric].values[0]
    df["delta_vs_baseline"] = df[metric] - baseline_val

    if std_col in df.columns:
        df["meaningful_gain"] = df["delta_vs_baseline"] > df[std_col]
    else:
        df["meaningful_gain"] = df["delta_vs_baseline"] > 0

    n_meaningful = df["meaningful_gain"].sum()
    log.info(
        "Decision rule (%s vs baseline=%s): %d / %d configs show meaningful gain "
        "(Î” > std).",
        metric, baseline_id, n_meaningful, len(df),
    )
    return df


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Summary table printer
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def format_summary_table(results_df: pd.DataFrame) -> str:
    """
    Format a human-readable ranked summary table.

    Returns
    -------
    str â€” multi-line table suitable for printing or saving as .txt
    """
    cols_to_show = []
    for c in [
        "config_id",
        "spearman_mean", "spearman_std",
        "rmse_mean",     "rmse_std",
        "ef1pct_mean",   "ef1pct_std",
        "calibration_mean",
        "delta_vs_baseline",
        "meaningful_gain",
    ]:
        if c in results_df.columns:
            cols_to_show.append(c)

    df = results_df.sort_values("spearman_mean", ascending=False) \
        if "spearman_mean" in results_df.columns else results_df

    lines = [
        "",
        "=" * 100,
        "  KINASEâ€“LIGAND PREDICTION â€” EXPERIMENT RESULTS",
        "  Ranked by Spearman Ï (mean across seeds) â†“",
        "=" * 100,
    ]

    # Header
    header_map = {
        "config_id":       "Config",
        "spearman_mean":   "Ï_mean",
        "spearman_std":    "Ï_std",
        "rmse_mean":       "RMSE_mean",
        "rmse_std":        "RMSE_std",
        "ef1pct_mean":     "EF1%_mean",
        "ef1pct_std":      "EF1%_std",
        "calibration_mean": "Cal_Ï",
        "delta_vs_baseline": "Î”_baseline",
        "meaningful_gain": "Meaningful?",
    }

    for i, (_, row) in enumerate(df.iterrows(), 1):
        lines.append(f"\n  {i:2d}. {row.get('config_id', 'N/A')}")
        for col in cols_to_show[1:]:
            val = row.get(col, "N/A")
            label = header_map.get(col, col)
            if isinstance(val, float):
                lines.append(f"       {label:<20s} = {val:+.4f}")
            else:
                lines.append(f"       {label:<20s} = {val}")

    lines.append("\n" + "=" * 100)
    lines.append(
        "  NOTE: EF1% marked as 'proxy' if <5% inactive background exists."
    )
    lines.append(
        "  Meaningful gain = True only if Î” > seed-std (decision rule applied)."
    )
    lines.append("=" * 100 + "\n")

    return "\n".join(lines)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Main experiment runner
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_all_experiments(
    dataset_path:  str,
    lig_store_path: str,
    prot_store_path: str,
    exp_cfg:       ExperimentConfig,
) -> dict:
    """
    Full experiment pipeline: train â†’ evaluate â†’ aggregate â†’ compare.

    Parameters
    ----------
    dataset_path    : path to dataset_clean.parquet
    lig_store_path  : path to ligand_features.pt
    prot_store_path : path to protein_features.pt
    exp_cfg         : ExperimentConfig

    Returns
    -------
    dict with:
        "config_agg_results" : {config_id: agg_dict}
        "results_df"         : pd.DataFrame (one row per config)
        "pairwise_df"        : pd.DataFrame (Wilcoxon tests)
    """
    results_dir = Path(exp_cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # â”€â”€ Load data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log.info("Loading dataset from %s â€¦", dataset_path)
    dataset_df = pd.read_parquet(dataset_path)
    lig_store  = load_ligand_feature_store(lig_store_path)
    prot_store = load_protein_feature_store(prot_store_path)

    if exp_cfg.fast_debug:
        log.warning("FAST DEBUG MODE: truncating dataset and epochs.")
        dataset_df = dataset_df.head(200).copy()
        exp_cfg.train_cfg.epochs     = 2
        exp_cfg.train_cfg.patience   = 2
        exp_cfg.config_ids           = exp_cfg.config_ids[:3]
        exp_cfg.seeds                = exp_cfg.seeds[:1]

    dataset = KinaseLigandDataset(
        dataset_df, lig_store, prot_store,
        use_sample_weight=exp_cfg.train_cfg.use_sample_weight,
    )

    # â”€â”€ Training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not exp_cfg.skip_training:
        log.info("Starting training: %d configs Ã— %d seeds",
                 len(exp_cfg.config_ids), len(exp_cfg.seeds))
        exp_cfg.train_cfg.checkpoint_dir = exp_cfg.checkpoint_dir
        exp_cfg.train_cfg.seeds          = exp_cfg.seeds

        train_results = train_all_configs(
            dataset    = dataset,
            train_cfg  = exp_cfg.train_cfg,
            config_ids = exp_cfg.config_ids,
            device     = exp_cfg.device,
        )
        # Save training summary
        train_summary = {
            cid: {
                str(seed): {
                    "best_val_loss": sr.get("best_val_loss"),
                    "best_epoch":    sr.get("best_epoch"),
                    "checkpoint":    sr.get("checkpoint_path"),
                }
                for seed, sr in seed_dict.items()
                if isinstance(sr, dict) and "best_val_loss" in sr
            }
            for cid, seed_dict in train_results.items()
        }
        with open(results_dir / "training_summary.json", "w") as f:
            json.dump(train_summary, f, indent=2)
        log.info("Training summary saved â†’ %s/training_summary.json", results_dir)
    else:
        log.info("Skipping training (--skip-training); loading existing checkpoints.")

    # â”€â”€ Per-seed evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log.info("Evaluating all configs Ã— seeds â€¦")
    config_agg_results: dict[str, dict] = {}
    all_seed_rows: list[dict] = []

    for config_id in progress_iter(
        exp_cfg.config_ids,
        total=len(exp_cfg.config_ids),
        desc="Experiment configs",
        leave=True,
    ):
        seed_results: list[dict] = []

        for seed in progress_iter(
            exp_cfg.seeds,
            total=len(exp_cfg.seeds),
            desc=f"{config_id} eval seeds",
            leave=False,
        ):
            try:
                seed_metrics = evaluate_single_seed(
                    config_id      = config_id,
                    seed           = seed,
                    dataset        = dataset,
                    checkpoint_dir = exp_cfg.checkpoint_dir,
                    threshold      = exp_cfg.threshold,
                    device         = exp_cfg.device,
                )
                seed_results.append(seed_metrics)
                all_seed_rows.append(seed_metrics)
            except FileNotFoundError:
                log.warning(
                    "Checkpoint not found for %s seed=%d â€” skipping.", config_id, seed
                )
            except Exception as exc:
                log.error(
                    "Evaluation failed for %s seed=%d: %s",
                    config_id, seed, exc, exc_info=True,
                )

        if seed_results:
            config_agg_results[config_id] = aggregate_config_seeds(
                config_id, seed_results
            )
        else:
            config_agg_results[config_id] = {
                "config_id": config_id,
                "error": "No successful seed evaluations",
            }

    # â”€â”€ Save per-seed raw results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    per_seed_df = pd.DataFrame(all_seed_rows)
    per_seed_df.to_csv(results_dir / "per_seed_results.csv", index=False)
    log.info("Per-seed results â†’ %s/per_seed_results.csv", results_dir)

    # â”€â”€ Build results DataFrame â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    results_rows = [
        {k: v for k, v in agg.items() if not k.endswith("_values")}
        for agg in config_agg_results.values()
    ]
    results_df = pd.DataFrame(results_rows)

    # Apply decision rule vs. ligand_only baseline
    primary_metric = "spearman_mean" if "spearman_mean" in results_df.columns else "roc_auc_mean"
    primary_std = "spearman_std" if primary_metric == "spearman_mean" else "roc_auc_std"
    results_df = apply_decision_rule(
        results_df,
        baseline_id = "ligand_only",
        metric      = primary_metric,
        std_col     = primary_std,
    )

    # Sort by Spearman descending
    if primary_metric in results_df.columns:
        results_df = results_df.sort_values(
            primary_metric, ascending=False
        ).reset_index(drop=True)

    results_csv_path = results_dir / "results.csv"
    results_df.to_csv(results_csv_path, index=False)
    log.info("Results saved â†’ %s", results_csv_path)

    # â”€â”€ Ensemble uncertainty + calibration for the best config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    best_config_id = results_df.iloc[0]["config_id"] \
        if not results_df.empty else exp_cfg.config_ids[0]
    log.info("Running ensemble uncertainty for best config: %s", best_config_id)

    try:
        if get_model_config(best_config_id).task_type != "regression":
            raise RuntimeError("Skipping uncertainty step for classification config.")
        ensemble = load_ensemble(
            best_config_id, exp_cfg.seeds, exp_cfg.checkpoint_dir, exp_cfg.device
        )
        from torch.utils.data import DataLoader, Subset
        test_idx = load_saved_split_indices(best_config_id, exp_cfg.seeds[0], exp_cfg.checkpoint_dir)
        test_ds = Subset(dataset, test_idx)
        test_loader = DataLoader(
            test_ds, batch_size=128, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )
        unc_result = ensemble.predict_loader(test_loader)

        # Calibration
        cal_metrics = compute_calibration_metrics(unc_result)
        with open(results_dir / "calibration_metrics.json", "w") as f:
            json.dump({k: v for k, v in cal_metrics.items()
                       if isinstance(v, (int, float))}, f, indent=2)
        log.info("Calibration metrics â†’ %s/calibration_metrics.json", results_dir)

        # Reliability diagram
        rel_df = reliability_diagram_data(unc_result)
        rel_df.to_csv(results_dir / "reliability_diagram.csv", index=False)
        log.info("Reliability diagram â†’ %s/reliability_diagram.csv", results_dir)

        # Uncertainty predictions
        unc_df = unc_result.to_dataframe()
        unc_df.to_csv(results_dir / "uncertainty_predictions.csv", index=False)
        log.info("Uncertainty predictions â†’ %s/uncertainty_predictions.csv", results_dir)

    except Exception as exc:
        log.error("Ensemble uncertainty step failed: %s", exc, exc_info=True)

    # â”€â”€ Pairwise Wilcoxon tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    pairwise_df = pd.DataFrame()
    if exp_cfg.run_pairwise and len(exp_cfg.config_ids) > 1:
        pairwise_df = pairwise_wilcoxon(
            config_agg_results,
            metric="spearman",
            alpha=0.05,
        )
        pairwise_df.to_csv(results_dir / "pairwise_significance.csv", index=False)
        log.info("Pairwise Wilcoxon â†’ %s/pairwise_significance.csv", results_dir)

    # â”€â”€ Human-readable summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    summary_text = format_summary_table(results_df)
    summary_path = results_dir / "experiment_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(summary_text)
    log.info("Summary â†’ %s", summary_path)

    # â”€â”€ Timing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    elapsed = time.time() - t_start
    log.info(
        "=" * 65 + "\n"
        "  Experiment complete in %.1f minutes.\n"
        "  Best config: %s  (Spearman=%.4f Â± %.4f)\n"
        "=" * 65,
        elapsed / 60,
        results_df.iloc[0]["config_id"] if not results_df.empty else "N/A",
        results_df.iloc[0].get(primary_metric, float("nan")) if not results_df.empty else float("nan"),
        results_df.iloc[0].get(primary_std,  float("nan")) if not results_df.empty else float("nan"),
    )

    return {
        "config_agg_results": config_agg_results,
        "results_df":         results_df,
        "pairwise_df":        pairwise_df,
    }


def run_time_bounded_ablation(
    dataset_path: str,
    lig_store_path: str,
    prot_store_path: str,
    exp_cfg: ExperimentConfig,
    time_limit_hours: float = 24.0,
) -> dict:
    results_dir = Path(exp_cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    budget_seconds = time_limit_hours * 3600
    config_ids = [cid for cid in PRIORITY_CONFIGS if cid in exp_cfg.config_ids]
    run_rows: list[dict] = []
    regression_run_times: list[float] = []
    hard_cls_run_times: list[float] = []
    posthoc_run_times: list[float] = []
    completed_regression: list[tuple[str, int]] = []

    print(
        f"[START] start_time={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))} "
        f"budget={_format_hms(budget_seconds)} configs={len(config_ids)} families=3"
    )

    dataset_df = pd.read_parquet(dataset_path)
    lig_store = load_ligand_feature_store(lig_store_path)
    prot_store = load_protein_feature_store(prot_store_path)
    base_dataset = KinaseLigandDataset(
        dataset_df,
        lig_store,
        prot_store,
        use_sample_weight=exp_cfg.train_cfg.use_sample_weight,
    )

    for family in ["regression", "hard_classification"]:
        family_train_cfg = _make_family_train_cfg(
            exp_cfg.train_cfg,
            family,
            exp_cfg.checkpoint_dir,
            seeds=[42],
            start_time=start_time,
            limit_hours=time_limit_hours,
        )
        family_stats = _family_dataset_stats(base_dataset.df, family, family_train_cfg)
        log.info(
            "[DATA] family=%s n_samples=%d dropped=%d active=%d inactive=%d unique_ligands=%d unique_targets=%d",
            family,
            family_stats["n_samples"],
            family_stats["dropped_rows"],
            family_stats["n_active"],
            family_stats["n_inactive"],
            family_stats["n_unique_ligands"],
            family_stats["n_unique_targets"],
        )

        for config_id in config_ids:
            history = regression_run_times if family == "regression" else hard_cls_run_times
            est = _estimate_next_run_seconds(history, fallback_hours=1.8 if family == "regression" else 1.0)
            elapsed = time.time() - start_time
            remaining = _remaining_seconds(start_time, time_limit_hours)
            if history and est > remaining:
                log.warning(
                    "Skipping family=%s config=%s seed=42 because next_run_estimate=%s exceeds remaining budget=%s",
                    family, config_id, _format_hms(est), _format_hms(remaining),
                )
                continue

            family_ckpt_dir = _family_checkpoint_dir(exp_cfg.checkpoint_dir, family)
            ckpt_path = Path(family_ckpt_dir) / f"{config_id}_seed42.pt"
            print(
                f"[START] family={family} config={config_id} seed=42 "
                f"elapsed={_format_hms(elapsed)} budget={_format_hms(budget_seconds)} "
                f"remaining_estimate={_format_hms(remaining)} next_run_estimate={_format_hms(est)}"
            )

            run_start = time.time()
            status = "completed"
            error = ""
            run_result = {}
            if ckpt_path.exists():
                status = "reused_checkpoint"
                log.info("Reusing checkpoint for family=%s config=%s seed=42 â†’ %s", family, config_id, ckpt_path)
            else:
                try:
                    run_result = train_config(config_id, base_dataset, family_train_cfg, device=exp_cfg.device)
                except Exception as exc:
                    status = "failed"
                    error = str(exc)
                    log.error("Run failed for family=%s config=%s: %s", family, config_id, exc, exc_info=True)

            train_time = time.time() - run_start
            if status != "failed":
                try:
                    eval_start = time.time()
                    seed_metrics = evaluate_single_seed(
                        config_id=config_id,
                        seed=42,
                        dataset=base_dataset,
                        checkpoint_dir=family_ckpt_dir,
                        threshold=exp_cfg.threshold,
                        device=exp_cfg.device,
                    )
                    eval_time = time.time() - eval_start
                    model, _ = load_checkpoint(config_id, 42, family_ckpt_dir, exp_cfg.device)
                    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    row = {
                        "family": family,
                        "config_id": config_id,
                        "task_type": model.cfg.task_type,
                        "label_scheme": model.cfg.label_scheme,
                        "seed": 42,
                        "checkpoint_path": str(Path(family_ckpt_dir) / f"{config_id}_seed42.pt"),
                        "train_time": train_time,
                        "eval_time": eval_time,
                        "status": status,
                        "error": error,
                        "dropped_rows": family_stats["dropped_rows"],
                        "split_sizes": "",
                        "n_params": n_params,
                        **seed_metrics,
                    }
                    run_rows.append(row)
                    if family == "regression":
                        completed_regression.append((config_id, 42))
                        regression_run_times.append(train_time + eval_time)
                    else:
                        hard_cls_run_times.append(train_time + eval_time)
                except Exception as exc:
                    log.error("Evaluation failed for family=%s config=%s: %s", family, config_id, exc, exc_info=True)
                    run_rows.append({
                        "family": family,
                        "config_id": config_id,
                        "task_type": "classification" if family != "regression" else "regression",
                        "label_scheme": family_train_cfg.label_scheme,
                        "seed": 42,
                        "checkpoint_path": str(ckpt_path),
                        "train_time": train_time,
                        "eval_time": 0.0,
                        "status": "failed",
                        "error": str(exc),
                        "dropped_rows": family_stats["dropped_rows"],
                        "split_sizes": "",
                        "n_params": np.nan,
                    })
            else:
                run_rows.append({
                    "family": family,
                    "config_id": config_id,
                    "task_type": "classification" if family != "regression" else "regression",
                    "label_scheme": family_train_cfg.label_scheme,
                    "seed": 42,
                    "checkpoint_path": str(ckpt_path),
                    "train_time": train_time,
                    "eval_time": 0.0,
                    "status": status,
                    "error": error,
                    "dropped_rows": family_stats["dropped_rows"],
                    "split_sizes": "",
                    "n_params": np.nan,
                })

            _flush_partial_results(results_dir, run_rows)
            total_elapsed = time.time() - start_time
            remaining = _remaining_seconds(start_time, time_limit_hours)
            next_est = _estimate_next_run_seconds(history, fallback_hours=1.8 if family == "regression" else 1.0)
            print(
                f"[END] family={family} config={config_id} seed=42 "
                f"duration={_format_hms(time.time() - run_start)} total_elapsed={_format_hms(total_elapsed)}"
            )
            print(
                f"[ETA] completed={len(run_rows)} remaining_estimate={_format_hms(remaining)} "
                f"next_run_estimate={_format_hms(next_est)}"
            )
            if total_elapsed >= budget_seconds:
                log.warning("%.1f-hour budget reached. Stopping cleanly.", time_limit_hours)
                per_seed_df, results_df = _flush_partial_results(results_dir, run_rows)
                return {"per_seed_df": per_seed_df, "results_df": results_df}

    extra_seed_candidates = [
        "full_model",
        "cross_attention",
        "ligand_plus_protein",
    ]
    est_extra = _estimate_next_run_seconds(regression_run_times, fallback_hours=1.8) * len(extra_seed_candidates) * 1.5
    if _remaining_seconds(start_time, time_limit_hours) > est_extra:
        extra_cfg = _make_family_train_cfg(
            exp_cfg.train_cfg,
            "regression",
            exp_cfg.checkpoint_dir,
            seeds=[52],
            start_time=start_time,
            limit_hours=time_limit_hours,
        )
        for config_id in extra_seed_candidates:
            family_ckpt_dir = _family_checkpoint_dir(exp_cfg.checkpoint_dir, "regression")
            ckpt_path = Path(family_ckpt_dir) / f"{config_id}_seed52.pt"
            if ckpt_path.exists():
                continue
            try:
                print(
                    f"[START] family=regression config={config_id} seed=52 "
                    f"elapsed={_format_hms(time.time()-start_time)} budget={_format_hms(budget_seconds)}"
                )
                run_start = time.time()
                train_config(config_id, base_dataset, extra_cfg, device=exp_cfg.device)
                eval_start = time.time()
                seed_metrics = evaluate_single_seed(
                    config_id=config_id,
                    seed=52,
                    dataset=base_dataset,
                    checkpoint_dir=family_ckpt_dir,
                    threshold=exp_cfg.threshold,
                    device=exp_cfg.device,
                )
                eval_time = time.time() - eval_start
                model, _ = load_checkpoint(config_id, 52, family_ckpt_dir, exp_cfg.device)
                run_rows.append({
                    "family": "regression",
                    "config_id": config_id,
                    "task_type": model.cfg.task_type,
                    "label_scheme": model.cfg.label_scheme,
                    "seed": 52,
                    "checkpoint_path": str(ckpt_path),
                    "train_time": time.time() - run_start,
                    "eval_time": eval_time,
                    "status": "completed",
                    "error": "",
                    "dropped_rows": 0,
                    "split_sizes": "",
                    "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    **seed_metrics,
                })
                regression_run_times.append((time.time() - run_start) + eval_time)
                _flush_partial_results(results_dir, run_rows)
            except Exception as exc:
                log.error("Optional seed 52 failed for %s: %s", config_id, exc, exc_info=True)

    posthoc_cfg = copy.deepcopy(exp_cfg.train_cfg)
    posthoc_stats = _family_dataset_stats(base_dataset.df, "posthoc", posthoc_cfg)
    log.info(
        "[DATA] family=posthoc n_samples=%d dropped=%d active=%d inactive=%d unique_ligands=%d unique_targets=%d",
        posthoc_stats["n_samples"],
        posthoc_stats["dropped_rows"],
        posthoc_stats["n_active"],
        posthoc_stats["n_inactive"],
        posthoc_stats["n_unique_ligands"],
        posthoc_stats["n_unique_targets"],
    )
    for config_id, seed in completed_regression:
        remaining = _remaining_seconds(start_time, time_limit_hours)
        est = _estimate_next_run_seconds(posthoc_run_times, fallback_hours=0.1)
        if posthoc_run_times and est > remaining:
            log.warning("Skipping posthoc family for %s seed=%d due to budget.", config_id, seed)
            continue
        try:
            print(
                f"[START] family=posthoc config={config_id} seed={seed} "
                f"elapsed={_format_hms(time.time()-start_time)} budget={_format_hms(budget_seconds)}"
            )
            eval_start = time.time()
            metrics = evaluate_posthoc_classification(
                config_id=config_id,
                seed=seed,
                dataset=base_dataset,
                checkpoint_dir=_family_checkpoint_dir(exp_cfg.checkpoint_dir, "regression"),
                threshold=exp_cfg.threshold,
                device=exp_cfg.device,
            )
            eval_time = time.time() - eval_start
            posthoc_run_times.append(eval_time)
            run_rows.append({
                "family": "posthoc",
                "config_id": config_id,
                "task_type": "posthoc",
                "label_scheme": "derived_from_regression",
                "seed": seed,
                "checkpoint_path": str(Path(_family_checkpoint_dir(exp_cfg.checkpoint_dir, "regression")) / f"{config_id}_seed{seed}.pt"),
                "train_time": 0.0,
                "eval_time": eval_time,
                "status": "completed",
                "error": "",
                "dropped_rows": posthoc_stats["dropped_rows"],
                "split_sizes": "",
                "n_params": np.nan,
                **metrics,
            })
            _flush_partial_results(results_dir, run_rows)
        except Exception as exc:
            log.error("Posthoc evaluation failed for %s seed=%d: %s", config_id, seed, exc, exc_info=True)

    regression_rows = [r for r in run_rows if r.get("family") == "regression" and r.get("status") in {"completed", "reused_checkpoint"}]
    if regression_rows:
        regression_df = pd.DataFrame(regression_rows)
        metric = "spearman"
        best_row = regression_df.sort_values(metric, ascending=False).iloc[0]
        best_config = str(best_row["config_id"])
        available_seeds = sorted(regression_df.loc[regression_df["config_id"] == best_config, "seed"].unique().tolist())
        try:
            ensemble = load_ensemble(
                best_config,
                available_seeds,
                _family_checkpoint_dir(exp_cfg.checkpoint_dir, "regression"),
                exp_cfg.device,
            )
            from torch.utils.data import DataLoader, Subset
            test_idx = load_saved_split_indices(best_config, available_seeds[0], _family_checkpoint_dir(exp_cfg.checkpoint_dir, "regression"))
            test_ds = Subset(base_dataset, test_idx)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0)
            unc_result = ensemble.predict_loader(test_loader)
            cal_metrics = compute_calibration_metrics(unc_result)
            with open(results_dir / "calibration_metrics.json", "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in cal_metrics.items() if isinstance(v, (int, float))}, f, indent=2)
            reliability_diagram_data(unc_result).to_csv(results_dir / "reliability_diagram.csv", index=False)
            unc_result.to_dataframe().to_csv(results_dir / "uncertainty_predictions.csv", index=False)
        except Exception as exc:
            log.error("Could not build regression calibration artifacts: %s", exc, exc_info=True)

    pairwise_frames = []
    if run_rows:
        raw_df = pd.DataFrame(run_rows)
        for family, fam_df in raw_df.groupby("family"):
            if fam_df["config_id"].nunique() < 2:
                continue
            metric = "spearman" if family == "regression" else "roc_auc"
            config_agg = {}
            for config_id, group in fam_df.groupby("config_id"):
                vals = group[metric].dropna().astype(float).tolist() if metric in group.columns else []
                config_agg[config_id] = {f"{metric}_values": vals, f"{metric}_mean": float(np.mean(vals)) if vals else float("nan")}
            fam_pairwise = pairwise_wilcoxon(config_agg, metric=metric, alpha=0.05)
            if not fam_pairwise.empty:
                fam_pairwise["family"] = family
                pairwise_frames.append(fam_pairwise)
    if pairwise_frames:
        pd.concat(pairwise_frames, ignore_index=True).to_csv(results_dir / "pairwise_significance.csv", index=False)

    per_seed_df, results_df = _flush_partial_results(results_dir, run_rows)
    return {"per_seed_df": per_seed_df, "results_df": results_df}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Entry point
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run all kinaseâ€“ligand experiments."
    )
    parser.add_argument("--dataset",     default="./pipeline_outputs/dataset_clean.parquet")
    parser.add_argument("--lig-store",   default="./pipeline_outputs/ligand_features")
    parser.add_argument("--prot-store",  default="./pipeline_outputs/protein_feature_store.pt")
    parser.add_argument("--config",      default="all",
                        help="Config ID(s) comma-separated, or 'all'")
    parser.add_argument("--seeds",       nargs="+", type=int, default=[42])
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--batch-size",  type=int, default=64 if torch.cuda.is_available() else 24)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--patience",    type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--time-limit-hours", type=float, default=24.0)
    parser.add_argument("--device",      default=DEVICE)
    parser.add_argument("--results-dir", default="./pipeline_outputs/results")
    parser.add_argument("--ckpt-dir",    default="./pipeline_outputs/checkpoints")
    parser.add_argument("--threshold",   type=float, default=7.0)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--no-pairwise",   action="store_true")
    parser.add_argument("--fast",          action="store_true",
                        help="Fast debug mode: 200 samples, 2 epochs, 3 configs")
    args = parser.parse_args()

    # Parse config list
    if args.config == "all":
        config_ids = PRIORITY_CONFIGS
    else:
        config_ids = [c.strip() for c in args.config.split(",")]

    train_cfg = TrainConfig(
        seeds          = args.seeds,
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        lr             = args.lr,
        patience       = args.patience,
        num_workers    = args.num_workers,
        checkpoint_dir = args.ckpt_dir,
    )

    exp_cfg = ExperimentConfig(
        config_ids     = config_ids,
        seeds          = args.seeds,
        train_cfg      = train_cfg,
        checkpoint_dir = args.ckpt_dir,
        results_dir    = args.results_dir,
        threshold      = args.threshold,
        device         = args.device,
        skip_training  = args.skip_training,
        run_pairwise   = not args.no_pairwise,
        fast_debug     = args.fast,
    )

    run_time_bounded_ablation(
        dataset_path    = args.dataset,
        lig_store_path  = args.lig_store,
        prot_store_path = args.prot_store,
        exp_cfg         = exp_cfg,
        time_limit_hours= args.time_limit_hours,
    )

