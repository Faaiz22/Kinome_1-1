"""
module8_evaluation.py
=====================
TITLE
Evaluation and reporting metrics for trained kinase-ligand models.

PURPOSE
This module measures model quality with explicit metric contracts so reported
performance can be compared, audited, and exported safely.

WHAT IT DOES
- Computes regression and classification metrics.
- Builds per-config and per-kinase evaluation outputs.
- Supports calibration and scaffold-aware analysis.
- Writes aggregated `results.csv` style tables.

HOW IT WORKS
1. Load saved checkpoints and matching test split indices.
2. Run prediction on aligned evaluation data.
3. Compute metric summaries and per-kinase tables.
4. Aggregate across seeds and write tidy outputs.

INPUT CONTRACT
- Aligned dataset plus ligand/protein feature stores.
- Existing checkpoints and saved split indices.

OUTPUT CONTRACT
- Flat metric dictionaries, per-kinase DataFrames, and results CSV tables.

DEPENDENCIES
- numpy, pandas, scipy, sklearn, torch
- module6_training.py
- module7_uncertainty.py

CRITICAL ASSUMPTIONS
- Test split indices correspond to the same training run.
- Evaluation data has already been feature-aligned.

FAILURE MODES
- Missing checkpoints or split files
- Invalid metric inputs
- Empty per-kinase subsets

SAFETY CHECKS IMPLEMENTED
- Config-aware evaluation branching
- Ensemble calibration only for regression tasks
- Graceful handling of empty per-kinase outputs

HOW TO RUN
- `python module8_evaluation.py --dataset ./pipeline_outputs/dataset_clean.parquet --config full_model`

HOW IT CONNECTS TO PIPELINE
It consumes trained checkpoints and aligned features to produce the metrics used
by experiments, exporters, and production diagnostics.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from progress_utils import progress_iter

log = logging.getLogger("module8")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Core metric functions
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation. Returns NaN if fewer than 3 samples."""
    if len(y_true) < 3:
        return float("nan")
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation. Returns NaN if fewer than 3 samples."""
    if len(y_true) < 3:
        return float("nan")
    r, _ = pearsonr(y_true, y_pred)
    return float(r)


def compute_ef1_percent(
    y_true:    np.ndarray,
    y_scores:  np.ndarray,
    threshold: float = 7.0,
    top_frac:  float = 0.01,
) -> tuple[float, bool]:
    """
    Compute Enrichment Factor at 1% (EF1%).

    Parameters
    ----------
    y_true    : true pIC50 values
    y_scores  : predicted scores (higher = more active)
    threshold : pIC50 cutoff defining "active" (default 7.0 = 10 nM IC50)
    top_frac  : fraction of the ranked list to consider (default 0.01 = 1%)

    Returns
    -------
    (ef1, is_valid_vs_metric)
        ef1                 : float  EF1% value
        is_valid_vs_metric  : bool   True if a genuine inactive background exists
    """
    N = len(y_true)
    if N == 0:
        return float("nan"), False

    n_top = max(1, int(np.ceil(N * top_frac)))

    # Sort by score descending
    ranked_idx    = np.argsort(y_scores)[::-1]
    y_true_ranked = y_true[ranked_idx]

    # Global hit rate
    actives_total = np.sum(y_true >= threshold)
    if actives_total == 0:
        log.warning("No actives at threshold=%.1f â€” EF1% is undefined.", threshold)
        return float("nan"), False

    global_hit_rate = actives_total / N

    # Fraction below threshold (inactive background fraction)
    inactive_frac = np.mean(y_true < threshold)

    # Validity check: need â‰¥5% inactives for a meaningful EF1%
    is_valid = inactive_frac >= 0.05

    if not is_valid:
        # â”€â”€ Ranking proxy EF1% â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Define "hit" as top-decile of true activity (relative cutoff)
        decile_threshold = np.percentile(y_true, 90)
        actives_in_top   = np.sum(y_true_ranked[:n_top] >= decile_threshold)
        n_actives_global = np.sum(y_true >= decile_threshold)

        if n_actives_global == 0:
            return float("nan"), False

        ef1 = (actives_in_top / n_top) / (n_actives_global / N)
        warnings.warn(
            f"EF1% computed as a RANKING PROXY (only {inactive_frac:.1%} "
            f"inactives at threshold {threshold:.1f}).  "
            f"This is NOT a valid virtual-screening metric without a "
            f"property-matched decoy background.",
            UserWarning, stacklevel=2,
        )
    else:
        # â”€â”€ True EF1% â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        hits_in_top = np.sum(y_true_ranked[:n_top] >= threshold)
        ef1 = (hits_in_top / n_top) / global_hit_rate

    return float(ef1), is_valid


def calibration_spearman(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    sigma:    np.ndarray,
) -> float:
    """
    Spearman correlation between absolute error and predicted uncertainty.

    A well-calibrated model should have Ï > 0: when the model says
    it is uncertain, it should actually be wrong more often.

    Returns
    -------
    float in [-1, 1]; NaN if fewer than 3 samples.
    """
    if len(y_true) < 3:
        return float("nan")
    abs_err = np.abs(y_true - y_pred)
    rho, _  = spearmanr(abs_err, sigma)
    return float(rho)


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    return metrics


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Full evaluation suite for a single prediction result
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def evaluate_predictions(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    sigma:      np.ndarray,
    uniprot_ids: list[str],
    inchikeys:   list[str],
    threshold:   float = 7.0,
    label:       str   = "test",
    classification_truth: Optional[np.ndarray] = None,
    classification_score: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute all evaluation metrics for a set of predictions.

    Parameters
    ----------
    y_true      : ground-truth pIC50 values
    y_pred      : predicted pIC50 values (ensemble mean)
    sigma       : predicted total uncertainty (std)
    uniprot_ids : list of UniProt IDs (one per sample)
    inchikeys   : list of InChIKeys (one per sample)
    threshold   : pIC50 threshold for EF1% active/inactive split
    label       : name for this evaluation set (used in log messages)

    Returns
    -------
    dict of metric name â†’ value
    """
    log.info("=" * 60)
    log.info("Evaluating: %s  (N=%d)", label, len(y_true))

    # â”€â”€ Global metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    rho   = spearman_rho(y_true, y_pred)
    rmse_ = rmse(y_true, y_pred)
    mae_  = mae(y_true, y_pred)
    pear  = pearson_r(y_true, y_pred)
    ef1, ef1_valid = compute_ef1_percent(y_true, y_pred, threshold)
    cal   = calibration_spearman(y_true, y_pred, sigma)

    log.info(
        "  Spearman Ï  = %.4f", rho
    )
    log.info(
        "  RMSE        = %.4f pIC50 units", rmse_
    )
    log.info(
        "  MAE         = %.4f pIC50 units", mae_
    )
    log.info(
        "  Pearson r   = %.4f", pear
    )
    log.info(
        "  EF1%%        = %.3f  [%s]",
        ef1,
        "valid VS metric" if ef1_valid else "RANKING PROXY ONLY"
    )
    log.info(
        "  Calibration = %.4f  (Spearman |err| vs Ïƒ)", cal
    )

    global_metrics = {
        f"{label}_spearman":         rho,
        f"{label}_rmse":             rmse_,
        f"{label}_mae":              mae_,
        f"{label}_pearson":          pear,
        f"{label}_ef1pct":           ef1,
        f"{label}_ef1pct_is_valid":  ef1_valid,
        f"{label}_calibration":      cal,
        f"{label}_n_samples":        len(y_true),
    }

    if classification_truth is None:
        classification_truth = (y_true >= threshold).astype(int)
    if classification_score is None:
        classification_score = y_pred

    cls_metrics = classification_metrics(classification_truth, classification_score)
    global_metrics.update({
        f"{label}_roc_auc": cls_metrics["roc_auc"],
        f"{label}_pr_auc": cls_metrics["pr_auc"],
        f"{label}_accuracy": cls_metrics["accuracy"],
        f"{label}_f1": cls_metrics["f1"],
    })

    # â”€â”€ Per-kinase evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    per_kinase = _per_kinase_metrics(y_true, y_pred, sigma, uniprot_ids)
    global_metrics["per_kinase"] = per_kinase

    log.info("Per-kinase Spearman Ï: mean=%.3f  median=%.3f  min=%.3f  max=%.3f",
             per_kinase["spearman_mean"],
             per_kinase["spearman_median"],
             per_kinase["spearman_min"],
             per_kinase["spearman_max"])

    return global_metrics


def _per_kinase_metrics(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    sigma:      np.ndarray,
    uniprot_ids: list[str],
) -> dict:
    """
    Compute per-kinase Spearman Ï, RMSE, and calibration.

    Only kinases with â‰¥ 3 data points are evaluated.

    Returns
    -------
    dict with aggregate statistics and per_kinase_df DataFrame.
    """
    uid_arr = np.array(uniprot_ids)
    rows    = []

    for uid in np.unique(uid_arr):
        mask = uid_arr == uid
        yt   = y_true[mask]
        yp   = y_pred[mask]
        sg   = sigma[mask]

        if len(yt) < 3:
            continue   # insufficient data for Spearman

        rho_k = spearman_rho(yt, yp)
        rmse_k = rmse(yt, yp)
        cal_k  = calibration_spearman(yt, yp, sg)

        rows.append({
            "uniprot_id": uid,
            "n_samples":  int(len(yt)),
            "spearman":   rho_k,
            "rmse":       rmse_k,
            "calibration": cal_k,
        })

    if not rows:
        return {
            "spearman_mean":   float("nan"),
            "spearman_median": float("nan"),
            "spearman_min":    float("nan"),
            "spearman_max":    float("nan"),
            "per_kinase_df":   pd.DataFrame(),
        }

    pk_df = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    spearman_vals = pk_df["spearman"].dropna().values

    return {
        "spearman_mean":   float(np.mean(spearman_vals)),
        "spearman_median": float(np.median(spearman_vals)),
        "spearman_min":    float(np.min(spearman_vals)),
        "spearman_max":    float(np.max(spearman_vals)),
        "per_kinase_df":   pk_df,
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Scaffold-stratified evaluation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def scaffold_stratified_evaluation(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    sigma:     np.ndarray,
    smiles_list: list[str],
    train_smiles: list[str],
    label:     str = "test",
) -> dict:
    """
    Evaluate separately on:
    - "novel_scaffold" : molecules whose Murcko scaffold does not appear in
                         the training portion of the test split
    - "seen_scaffold"  : molecules whose scaffold appears â‰¥ 1 time

    Requires smiles_list aligned with y_true/y_pred.

    Returns
    -------
    dict with metrics for each scaffold stratum.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            sc  = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) \
                  if mol else ""
        except Exception:
            sc = ""
        scaffolds.append(sc)

    # Compute train scaffolds
    train_scaffolds = set()
    for smi in train_smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            sc  = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) \
                  if mol else ""
        except Exception:
            sc = ""
        train_scaffolds.add(sc)

    scaffold_arr = np.array(scaffolds)

    # "Novel" scaffolds not in train
    novel_mask = np.array([sc not in train_scaffolds for sc in scaffolds])
    seen_mask  = ~novel_mask

    results = {}
    for stratum, mask in [("novel_scaffold", novel_mask), ("seen_scaffold", seen_mask)]:
        if mask.sum() < 3:
            log.info("Stratum '%s': only %d samples â€” skipping.", stratum, mask.sum())
            continue
        yt = y_true[mask]; yp = y_pred[mask]; sg = sigma[mask]
        results[stratum] = {
            "spearman": spearman_rho(yt, yp),
            "rmse":     rmse(yt, yp),
            "n":        int(mask.sum()),
        }
        log.info(
            "Scaffold stratum '%s': N=%d  Spearman=%.3f  RMSE=%.4f",
            stratum, mask.sum(), results[stratum]["spearman"], results[stratum]["rmse"],
        )
    return results


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Multi-seed aggregation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def aggregate_seed_metrics(
    seed_metrics: list[dict],
    scalar_keys:  Optional[list[str]] = None,
) -> dict:
    """
    Aggregate a list of per-seed metric dicts into mean Â± std.

    Parameters
    ----------
    seed_metrics : list of dicts (one per seed), each from evaluate_predictions()
    scalar_keys  : specific keys to aggregate; if None, aggregates all
                   top-level scalar float/int values

    Returns
    -------
    dict with keys like "{metric}_mean" and "{metric}_std"
    """
    if not seed_metrics:
        return {}

    all_keys = scalar_keys or [
        k for k, v in seed_metrics[0].items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]

    agg = {}
    for key in all_keys:
        vals = [m[key] for m in seed_metrics if key in m and not np.isnan(m.get(key, np.nan))]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"]  = float(np.std(vals, ddof=0))
        else:
            agg[f"{key}_mean"] = float("nan")
            agg[f"{key}_std"]  = float("nan")

    return agg


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Full evaluation pipeline (integrates with module7)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_full_evaluation(
    config_id:      str,
    seeds:          list[int],
    dataset_df:     pd.DataFrame,
    ligand_store:   dict,
    protein_store:  dict,
    checkpoint_dir: str   = "./checkpoints",
    threshold:      float = 7.0,
    device:         str   = "cpu",
) -> dict:
    """
    Full evaluation pipeline for one configuration across all seeds.

    Steps
    -----
    1. Load ensemble from checkpoints.
    2. Build test split (Murcko scaffold split, same split as training).
    3. Run ensemble prediction â†’ UncertaintyResult.
    4. Compute all evaluation metrics.
    5. Aggregate across seeds.

    Returns
    -------
    dict with all aggregated metrics, suitable for appending to results.csv
    """
    import torch
    from torch.utils.data import DataLoader, Subset

    from module6_training import (
        KinaseLigandDataset,
        collate_fn,
        load_saved_split_indices,
    )
    from module7_uncertainty import load_ensemble, compute_calibration_metrics
    from module5_models import get_model_config

    log.info("=" * 65)
    log.info("Full evaluation: config=%s  seeds=%s", config_id, seeds)
    probe_model, _ = load_checkpoint(config_id, seeds[0], checkpoint_dir, device)
    model_cfg = probe_model.cfg

    dataset = KinaseLigandDataset(dataset_df, ligand_store, protein_store,
                                  use_sample_weight=False)
    smiles_list = dataset.get_smiles_list()
    uniprot_all = dataset.get_uniprot_ids()

    # Load the same scaffold split used during training
    test_idx = load_saved_split_indices(
        config_id=config_id,
        seed=seeds[0],
        checkpoint_dir=checkpoint_dir,
    )
    all_indices = set(range(len(dataset)))
    train_idx = sorted(all_indices - set(test_idx))

    test_ds = Subset(dataset, test_idx)
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn,
        num_workers=0,
    )

    # Evaluate each seed independently
    seed_metrics = []
    for seed in progress_iter(
        seeds,
        total=len(seeds),
        desc=f"Eval {config_id} seeds",
        leave=True,
    ):
        model, _ = load_checkpoint(config_id, seed, checkpoint_dir, device)
        preds = predict(model, test_loader, device)
        y_true = preds["targets"]

        if model_cfg.task_type == "classification":
            y_score = preds["probs"]
            y_label = preds["labels"].astype(int)
            cls = classification_metrics(y_label, y_score)
            seed_metrics.append({
                "roc_auc": cls["roc_auc"],
                "pr_auc": cls["pr_auc"],
                "accuracy": cls["accuracy"],
                "f1": cls["f1"],
                "n_samples": len(y_label),
            })
        else:
            y_pred = preds["mu"]
            sigma = np.sqrt(np.clip(preds["var"], 1e-8, None))

            rho = spearman_rho(y_true, y_pred)
            rmse_ = rmse(y_true, y_pred)
            ef1, ef1_valid = compute_ef1_percent(y_true, y_pred, threshold)
            cal = calibration_spearman(y_true, y_pred, sigma)
            cls = classification_metrics((y_true >= threshold).astype(int), y_pred)

            seed_metrics.append({
                "spearman": rho,
                "rmse": rmse_,
                "ef1pct": ef1,
                "ef1pct_is_valid": ef1_valid,
                "calibration": cal,
                "roc_auc": cls["roc_auc"],
                "pr_auc": cls["pr_auc"],
                "accuracy": cls["accuracy"],
                "f1": cls["f1"],
                "n_samples": len(y_true),
            })

    # Aggregate across seeds
    global_metrics = aggregate_seed_metrics(seed_metrics)
    rename_map = {
        "spearman_mean": "test_spearman_mean",
        "spearman_std": "test_spearman_std",
        "rmse_mean": "test_rmse_mean",
        "rmse_std": "test_rmse_std",
        "mae_mean": "test_mae_mean",
        "mae_std": "test_mae_std",
        "pearson_mean": "test_pearson_mean",
        "pearson_std": "test_pearson_std",
        "ef1pct_mean": "test_ef1pct_mean",
        "ef1pct_std": "test_ef1pct_std",
        "calibration_mean": "test_calibration_mean",
        "calibration_std": "test_calibration_std",
        "n_samples_mean": "test_n_samples_mean",
        "n_samples_std": "test_n_samples_std",
    }
    global_metrics = {
        rename_map.get(k, k): v for k, v in global_metrics.items()
    }
    global_metrics["task_type"] = model_cfg.task_type
    if "ef1pct_is_valid" in seed_metrics[0]:
        global_metrics["test_ef1pct_is_valid"] = any(
            m["ef1pct_is_valid"] for m in seed_metrics
        )

    if model_cfg.task_type == "classification":
        # Use the first seed for detailed classification evaluation artifacts.
        model, _ = load_checkpoint(config_id, seeds[0], checkpoint_dir, device)
        preds = predict(model, test_loader, device)
        eval_metrics = {
            "per_kinase": {"per_kinase_df": pd.DataFrame()},
        }
        cal_metrics = {}
        scaffold_metrics = {}
        result = None
    else:
        # For calibration, use ensemble
        ensemble = load_ensemble(config_id, seeds, checkpoint_dir, device)
        result = ensemble.predict_loader(test_loader)
        cal_metrics = compute_calibration_metrics(result)
        eval_metrics = evaluate_predictions(
            y_true=result.targets,
            y_pred=result.pred_mean,
            sigma=result.pred_std,
            uniprot_ids=result.uniprot_ids,
            inchikeys=result.inchikeys,
            threshold=threshold,
            label="test",
        )

        # Scaffold-stratified evaluation
        test_smiles = [smiles_list[i] for i in test_idx]
        train_smiles = [smiles_list[i] for i in train_idx]
        scaffold_metrics = scaffold_stratified_evaluation(
            y_true      = result.targets,
            y_pred      = result.pred_mean,
            sigma       = result.pred_std,
            smiles_list = test_smiles,
            train_smiles = train_smiles,
            label       = config_id,
        )

    # Build flat results row (per-kinase DataFrame stored separately)
    per_kinase_df = eval_metrics.get("per_kinase", {}).get("per_kinase_df", pd.DataFrame())

    flat = {
        "config_id":        config_id,
        **{k: v for k, v in global_metrics.items() if not isinstance(v, dict)},
        **{f"cal_{k}": v for k, v in cal_metrics.items()
           if isinstance(v, (int, float))},
        **{f"scaffold_{stratum}_{met}": val
           for stratum, stratum_dict in scaffold_metrics.items()
           for met, val in stratum_dict.items()
           if isinstance(val, (int, float))},
        "n_test_samples":   result.n_samples() if result is not None else len(preds["targets"]),
        "n_ensemble_seeds": result.n_seeds() if result is not None else len(seeds),
    }

    log.info("Evaluation complete for %s.", config_id)
    return {
        "flat_metrics":   flat,
        "per_kinase_df":  per_kinase_df,
        "uncertainty_result": result,
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Results CSV builder
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_results_csv(
    all_eval_results: dict,    # {config_id: eval_result_dict}
    output_path: str = "results.csv",
) -> pd.DataFrame:
    """
    Aggregate per-config evaluation results into a tidy results.csv.

    The decision rule from the spec is applied:
        Reject an apparent improvement if Î”metric < std

    Parameters
    ----------
    all_eval_results : {config_id: dict from run_full_evaluation()}
    output_path      : where to save results.csv

    Returns
    -------
    pd.DataFrame with one row per config_id
    """
    rows = []
    for cid, eval_res in all_eval_results.items():
        if "error" in eval_res:
            rows.append({"config_id": cid, "error": eval_res["error"]})
            continue
        flat = eval_res.get("flat_metrics", {})
        rows.append(flat)

    df = pd.DataFrame(rows)

    # Compute Î”-spearman and flag statistically meaningful improvements
    if "test_spearman_mean" in df.columns and "test_spearman_std" in df.columns:
        baseline = df[df["config_id"] == "ligand_only"]["test_spearman_mean"].values
        if len(baseline) > 0:
            b = baseline[0]
            b_std = df[df["config_id"] == "ligand_only"]["test_spearman_std"].values
            b_std = b_std[0] if len(b_std) else 0.0
            df["delta_spearman"]   = df["test_spearman_mean"] - b
            df["meaningful_gain"]  = df["delta_spearman"] > df["test_spearman_std"]

    df.to_csv(output_path, index=False)
    log.info("Results saved â†’ %s  (%d configs)", output_path, len(df))
    return df


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Per-kinase results writer
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def save_per_kinase_results(
    all_eval_results: dict,
    output_path: str = "per_kinase_results.csv",
) -> None:
    """Save concatenated per-kinase metrics across all configs."""
    frames = []
    for cid, eval_res in all_eval_results.items():
        if "per_kinase_df" not in eval_res:
            continue
        pk_df = eval_res["per_kinase_df"].copy()
        if pk_df.empty:
            continue
        pk_df["config_id"] = cid
        frames.append(pk_df)

    if not frames:
        log.warning("No per-kinase data to save.")
        return

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(output_path, index=False)
    log.info("Per-kinase results â†’ %s", output_path)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Entry point
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    import argparse
    import torch
    from module2_feature_engineering import load_ligand_feature_store
    from module3_protein_features import load_protein_feature_store
    from module5_models import ALL_CONFIG_IDS

    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument("--dataset",    default="./pipeline_outputs/dataset_clean.parquet")
    parser.add_argument("--lig-store",  default="./pipeline_outputs/ligand_features")
    parser.add_argument("--prot-store", default="./pipeline_outputs/protein_feature_store.pt")
    parser.add_argument("--config",     default="full_model",
                        help="Config ID or 'all'")
    parser.add_argument("--seeds",      nargs="+", type=int, default=[42])
    parser.add_argument("--ckpt-dir",   default="./pipeline_outputs/checkpoints")
    parser.add_argument("--threshold",  type=float, default=7.0)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--output",     default="./pipeline_outputs/results/results.csv")
    args = parser.parse_args()

    df         = pd.read_parquet(args.dataset)
    lig_store  = load_ligand_feature_store(args.lig_store)
    prot_store = load_protein_feature_store(args.prot_store)

    config_ids = ALL_CONFIG_IDS if args.config == "all" else [args.config]
    all_eval   = {}

    for cid in config_ids:
        try:
            all_eval[cid] = run_full_evaluation(
                config_id      = cid,
                seeds          = args.seeds,
                dataset_df     = df,
                ligand_store   = lig_store,
                protein_store  = prot_store,
                checkpoint_dir = args.ckpt_dir,
                threshold      = args.threshold,
                device         = args.device,
            )
        except Exception as e:
            log.error("Evaluation failed for %s: %s", cid, e, exc_info=True)
            all_eval[cid] = {"error": str(e)}

    results_df = build_results_csv(all_eval, output_path=args.output)
    save_per_kinase_results(all_eval, output_path="per_kinase_results.csv")

    print("\nâ”€â”€ Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    display_cols = [c for c in [
        "config_id",
        "test_spearman_mean", "test_spearman_std",
        "test_rmse_mean",     "test_rmse_std",
        "test_ef1pct_mean",   "test_ef1pct_std",
        "cal_spearman_err_sigma_mean",
    ] if c in results_df.columns]
    print(results_df[display_cols].to_string(index=False))

