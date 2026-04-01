"""
module7_uncertainty.py
======================
TITLE
Ensemble uncertainty estimation for kinase-ligand prediction.

PURPOSE
This module turns saved multi-seed model checkpoints into scientifically
interpretable uncertainty estimates rather than decorative confidence numbers.

WHAT IT DOES
- Loads ensembles from saved checkpoints.
- Computes aleatoric, epistemic, and total uncertainty.
- Produces confidence intervals and calibration-ready outputs.
- Supports ranked screening outputs with uncertainty filters.

HOW IT WORKS
1. Load one model per saved seed.
2. Run aligned predictions across the same sample order.
3. Decompose variance into aleatoric and epistemic terms.
4. Export uncertainty tables and calibration inputs.

INPUT CONTRACT
- Regression checkpoints with consistent sample ordering.
- Data loaders or ligand-target panels aligned to the pipeline schema.

OUTPUT CONTRACT
- `UncertaintyResult` objects and derived DataFrames.
- Calibration and ranked-hit utilities.

DEPENDENCIES
- torch, numpy, pandas
- module6_training.py

CRITICAL ASSUMPTIONS
- Ensemble members correspond to the same model config.
- Prediction order is stable across seeds.

FAILURE MODES
- Missing checkpoints
- Mismatched prediction ordering across seeds
- Classification configs passed into regression uncertainty logic

SAFETY CHECKS IMPLEMENTED
- Empty-ensemble rejection
- Order-alignment assertions
- Explicit decomposition outputs and interval bounds

HOW TO RUN
- `python module7_uncertainty.py --config full_model --ckpt-dir ./pipeline_outputs/checkpoints`

HOW IT CONNECTS TO PIPELINE
It consumes module6 checkpoints and feeds uncertainty-aware evaluation,
experiment reporting, and Streamlit inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from module5_models import BaseModel
from module6_training import (
    load_checkpoint,
    predict,
    _to_device,
    ALL_CONFIG_IDS,
    DEVICE,
)

log = logging.getLogger("module7")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Per-sample uncertainty result
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@dataclass
class UncertaintyResult:
    """
    Uncertainty-decomposed predictions for a single dataset split.

    Attributes
    ----------
    inchikeys      : molecule identifiers
    uniprot_ids    : target identifiers
    targets        : ground-truth pIC50 values
    pred_mean      : ensemble mean prediction
    aleatoric_var  : mean predicted variance (aleatoric)
    epistemic_var  : variance of predicted means (epistemic)
    total_var      : aleatoric + epistemic
    pred_std       : sqrt(total_var)
    lower_95       : pred_mean - 1.96 * pred_std
    upper_95       : pred_mean + 1.96 * pred_std
    seed_mus       : (M, N) array of per-seed predictions
    seed_vars      : (M, N) array of per-seed predicted variances
    """
    inchikeys:      list[str]
    uniprot_ids:    list[str]
    targets:        np.ndarray
    pred_mean:      np.ndarray
    aleatoric_var:  np.ndarray
    epistemic_var:  np.ndarray
    total_var:      np.ndarray
    pred_std:       np.ndarray
    lower_95:       np.ndarray
    upper_95:       np.ndarray
    seed_mus:       np.ndarray          # (M, N)
    seed_vars:      np.ndarray          # (M, N)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "inchikey":     self.inchikeys,
            "uniprot_id":   self.uniprot_ids,
            "target":       self.targets,
            "pred_mean":    self.pred_mean,
            "pred_std":     self.pred_std,
            "aleatoric_std": np.sqrt(np.clip(self.aleatoric_var, 0, None)),
            "epistemic_std": np.sqrt(np.clip(self.epistemic_var, 0, None)),
            "lower_95":     self.lower_95,
            "upper_95":     self.upper_95,
        })

    def n_samples(self) -> int:
        return len(self.targets)

    def n_seeds(self) -> int:
        return self.seed_mus.shape[0]


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Ensemble uncertainty engine
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class UncertaintyEnsemble:
    """
    Compute ensemble predictions and decomposed uncertainties.

    Parameters
    ----------
    models : list of trained BaseModel instances (M models = M seeds)
    device : inference device
    """

    def __init__(
        self,
        models: list[BaseModel],
        device: str = DEVICE,
    ) -> None:
        if not models:
            raise ValueError("Provide at least one trained model.")
        self.models = [m.eval().to(device) for m in models]
        self.device = device
        log.info("UncertaintyEnsemble: %d member models", len(models))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_loader(self, loader: DataLoader) -> UncertaintyResult:
        """
        Run all ensemble members on *loader* and return decomposed uncertainty.

        Parameters
        ----------
        loader : DataLoader over a KinaseLigandDataset split

        Returns
        -------
        UncertaintyResult
        """
        if hasattr(loader, 'shuffle') and loader.shuffle:
            raise ValueError("loader.shuffle must be False for ensemble alignment")

        M = len(self.models)
        # Per-seed arrays (collected first pass)
        seed_mus:  list[np.ndarray] = []
        seed_vars: list[np.ndarray] = []
        # Meta (only need from first model)
        all_targets:  Optional[np.ndarray] = None
        all_inchikeys: Optional[list[str]] = None
        all_uniprots:  Optional[list[str]] = None

        for m_idx, model in enumerate(self.models):
            preds = predict(model, loader, self.device)
            seed_mus.append(preds["mu"])
            seed_vars.append(preds["var"])
            if m_idx == 0:
                all_targets   = preds["targets"]
                all_inchikeys = preds["inchikeys"]
                all_uniprots  = preds["uniprots"]
            else:
                assert preds["inchikeys"] == all_inchikeys, "InChIKey order mismatch across seeds"
                assert preds["uniprots"] == all_uniprots, "UniProt order mismatch across seeds"

        seed_mus_arr  = np.stack(seed_mus,  axis=0)   # (M, N)
        seed_vars_arr = np.stack(seed_vars, axis=0)   # (M, N)

        # â”€â”€ Decomposition â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        ensemble_mean   = seed_mus_arr.mean(axis=0)                  # (N,)
        aleatoric_var   = seed_vars_arr.mean(axis=0)                 # (N,)
        epistemic_var   = seed_mus_arr.var(axis=0, ddof=0)           # (N,)  Bessel's not applied: M is small
        total_var       = aleatoric_var + epistemic_var               # (N,)

        # Clamp to avoid sqrt of tiny negatives from floating-point noise
        total_var    = np.clip(total_var,    a_min=1e-8, a_max=None)
        aleatoric_var = np.clip(aleatoric_var, a_min=1e-8, a_max=None)
        epistemic_var = np.clip(epistemic_var, a_min=0.0,  a_max=None)

        pred_std = np.sqrt(total_var)
        lower_95 = ensemble_mean - 1.96 * pred_std
        upper_95 = ensemble_mean + 1.96 * pred_std

        log.info(
            "Ensemble predictions (N=%d, M=%d): "
            "mean_pred=%.3f  mean_Ïƒ_total=%.3f  "
            "mean_Ïƒ_al=%.3f  mean_Ïƒ_ep=%.3f",
            len(ensemble_mean), M,
            ensemble_mean.mean(),
            pred_std.mean(),
            np.sqrt(aleatoric_var).mean(),
            np.sqrt(epistemic_var).mean(),
        )

        return UncertaintyResult(
            inchikeys     = all_inchikeys,
            uniprot_ids   = all_uniprots,
            targets       = all_targets,
            pred_mean     = ensemble_mean,
            aleatoric_var = aleatoric_var,
            epistemic_var = epistemic_var,
            total_var     = total_var,
            pred_std      = pred_std,
            lower_95      = lower_95,
            upper_95      = upper_95,
            seed_mus      = seed_mus_arr,
            seed_vars     = seed_vars_arr,
        )

    # ------------------------------------------------------------------
    def predict_smiles_list(
        self,
        smiles_list:  list[str],
        ligand_store: dict,
        protein_store: dict,
        uniprot_id:   str,
        batch_size:   int = 64,
    ) -> UncertaintyResult:
        """
        Convenience wrapper: predict for an arbitrary list of SMILES strings
        against a single kinase target (useful for virtual screening).

        Parameters
        ----------
        smiles_list   : list of canonical SMILES to predict
        ligand_store  : {inchikey: Data} from module2
        protein_store : {uniprot_id: ProteinFeatures} from module3
        uniprot_id    : target kinase UniProt accession
        batch_size    : inference batch size
        """
        from module6_training import KinaseLigandDataset, collate_fn
        import pandas as pd
        from rdkit import Chem
        from rdkit.Chem import InchiToInchiKey, MolToInchi

        if uniprot_id not in protein_store:
            raise ValueError(f"UniProt {uniprot_id} not in protein_store.")

        # Build a minimal dataframe (targets are dummy â€” we only need predictions)
        rows = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            inchi = MolToInchi(mol)
            ik = InchiToInchiKey(inchi) if inchi else None
            if ik is None or ik not in ligand_store:
                log.warning("SMILES not in ligand_store (not featurised): %s", smi)
                continue
            rows.append({
                "inchikey":   ik,
                "smiles":     smi,
                "uniprot_id": uniprot_id,
                "pIC50":      7.0,       # dummy
                "pIC50_std":  0.5,       # dummy
            })

        if not rows:
            raise ValueError("No valid featurised SMILES for virtual screening.")

        df = pd.DataFrame(rows)
        vs_dataset = KinaseLigandDataset(df, ligand_store, protein_store,
                                         use_sample_weight=False)
        loader = DataLoader(
            vs_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn,
        )
        return self.predict_loader(loader)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Calibration metrics
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compute_calibration_metrics(result: UncertaintyResult) -> dict:
    """
    Compute uncertainty calibration quality metrics.

    Metrics
    -------
    spearman_err_sigma : Spearman Ï between |error| and Ïƒ_total
                         Higher is better; > 0 means the model knows
                         when it is uncertain.
    ece_95             : Empirical coverage of 95% prediction interval
                         Ideal = 0.95; > 0.95 = over-confident.
    ece_table          : coverage at 10%â€¦90% CI levels
    sharpness          : mean(Ïƒ_total)  â€” lower = sharper forecasts
    mean_nll           : mean Gaussian NLL on test set

    Returns
    -------
    dict of metric name â†’ value
    """
    from scipy.stats import spearmanr

    errors = np.abs(result.pred_mean - result.targets)
    sigmas = result.pred_std

    # Spearman correlation between |error| and Ïƒ
    rho, p_val = spearmanr(errors, sigmas)

    # Coverage at 95% CI
    in_95 = np.mean(
        (result.targets >= result.lower_95) &
        (result.targets <= result.upper_95)
    )

    # Coverage table at multiple levels
    coverage_table = {}
    for alpha in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]:
        z = _alpha_to_z(alpha)
        lo = result.pred_mean - z * sigmas
        hi = result.pred_mean + z * sigmas
        coverage_table[f"coverage_{int(alpha*100)}"] = float(
            np.mean((result.targets >= lo) & (result.targets <= hi))
        )

    # Expected Calibration Error (ECE) â€” mean absolute deviation from ideal coverage
    ece = float(np.mean([
        abs(v - float(k.split("_")[1]) / 100)
        for k, v in coverage_table.items()
    ]))

    # Sharpness (lower = sharper)
    sharpness = float(sigmas.mean())

    # Mean Gaussian NLL on predictions
    mean_nll = float(np.mean(
        0.5 * np.log(2 * np.pi * result.total_var) +
        0.5 * (result.targets - result.pred_mean) ** 2 / result.total_var
    ))

    metrics = {
        "spearman_err_sigma": float(rho),
        "spearman_p_value":   float(p_val),
        "coverage_95":        float(in_95),
        "ece":                ece,
        "sharpness":          sharpness,
        "mean_nll":           mean_nll,
        "mean_abs_error":     float(errors.mean()),
        **coverage_table,
    }

    log.info(
        "Calibration: Spearman(|err|,Ïƒ)=%.3f  Coverage@95=%.3f  ECE=%.3f  "
        "Sharpness=%.3f  NLL=%.3f",
        rho, in_95, ece, sharpness, mean_nll,
    )
    return metrics


def _alpha_to_z(alpha: float) -> float:
    """Convert a coverage probability to a standard normal z-score."""
    from scipy.stats import norm
    return float(norm.ppf(1 - (1 - alpha) / 2))


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Reliability diagram data (for plotting)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def reliability_diagram_data(
    result: UncertaintyResult,
    n_bins: int = 20,
) -> pd.DataFrame:
    """
    Compute observed-vs-expected coverage for a reliability diagram.

    Returns a DataFrame with columns:
        expected_coverage, observed_coverage, count
    """
    alphas  = np.linspace(0.05, 0.99, n_bins)
    rows    = []
    for alpha in alphas:
        z  = _alpha_to_z(float(alpha))
        lo = result.pred_mean - z * result.pred_std
        hi = result.pred_mean + z * result.pred_std
        observed = float(np.mean(
            (result.targets >= lo) & (result.targets <= hi)
        ))
        rows.append({
            "expected_coverage": float(alpha),
            "observed_coverage": observed,
            "count": int(np.sum(
                (result.targets >= lo) & (result.targets <= hi)
            )),
        })
    return pd.DataFrame(rows)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Load ensemble from checkpoints
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def load_ensemble(
    config_id:      str,
    seeds:          list[int],
    checkpoint_dir: str = "./checkpoints",
    device:         str = DEVICE,
) -> "UncertaintyEnsemble":
    """
    Load a trained ensemble for *config_id* from saved checkpoints.

    Parameters
    ----------
    config_id      : model configuration name
    seeds          : list of seeds (each â†’ one ensemble member)
    checkpoint_dir : directory containing .pt checkpoint files
    device         : inference device

    Returns
    -------
    UncertaintyEnsemble
    """
    models: list[BaseModel] = []
    for seed in seeds:
        model, _ = load_checkpoint(config_id, seed, checkpoint_dir, device)
        models.append(model)
    return UncertaintyEnsemble(models, device=device)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Uncertainty-ranked virtual screening
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def uncertainty_ranked_hits(
    result:       UncertaintyResult,
    top_k:        int   = 50,
    min_pred:     float = 7.0,    # minimum predicted pIC50
    max_epistemic_std: float = 1.0,  # reject highly uncertain predictions
) -> pd.DataFrame:
    """
    Rank virtual screening hits by predicted pIC50, filtered by uncertainty.

    Strategy
    --------
    1. Filter predictions where epistemic_std > max_epistemic_std
       (model is uncertain â€” likely out-of-distribution).
    2. Filter predictions where pred_mean < min_pred.
    3. Rank by pred_mean descending.
    4. Return top-k hits with uncertainty bounds.

    Returns a DataFrame suitable for hand-off to medicinal chemists.
    """
    df = result.to_dataframe()
    ep_std = np.sqrt(result.epistemic_var)

    df["epistemic_std"] = ep_std
    df["in_ci_95"] = (
        (result.targets >= result.lower_95) &
        (result.targets <= result.upper_95)
    )

    filtered = df[
        (df["pred_mean"] >= min_pred) &
        (df["epistemic_std"] <= max_epistemic_std)
    ].copy()

    filtered = filtered.sort_values("pred_mean", ascending=False).head(top_k)
    filtered["rank"] = range(1, len(filtered) + 1)

    log.info(
        "Uncertainty-filtered hits: %d / %d (top-%d shown, "
        "predâ‰¥%.1f, ep_stdâ‰¤%.2f)",
        len(filtered), len(df), top_k, min_pred, max_epistemic_std,
    )
    return filtered.reset_index(drop=True)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Entry point
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    import argparse
    from module2_feature_engineering import load_ligand_feature_store
    from module3_protein_features import load_protein_feature_store
    from module6_training import (
        KinaseLigandDataset, collate_fn, murcko_scaffold_split, TrainConfig
    )

    parser = argparse.ArgumentParser(description="Uncertainty quantification.")
    parser.add_argument("--dataset",    default="./pipeline_outputs/dataset_clean.parquet")
    parser.add_argument("--lig-store",  default="./pipeline_outputs/ligand_features")
    parser.add_argument("--prot-store", default="./pipeline_outputs/protein_feature_store.pt")
    parser.add_argument("--config",     default="full_model")
    parser.add_argument("--seeds",      nargs="+", type=int, default=[42])
    parser.add_argument("--ckpt-dir",   default="./pipeline_outputs/checkpoints")
    parser.add_argument("--device",     default=DEVICE)
    parser.add_argument("--output",     default="./pipeline_outputs/results/uncertainty_predictions.csv")
    args = parser.parse_args()

    import pandas as pd

    df         = pd.read_parquet(args.dataset)
    lig_store  = load_ligand_feature_store(args.lig_store)
    prot_store = load_protein_feature_store(args.prot_store)

    dataset = KinaseLigandDataset(df, lig_store, prot_store)

    smiles_list = dataset.get_smiles_list()
    _, _, test_idx = murcko_scaffold_split(
        smiles_list, val_frac=0.1, test_frac=0.1, seed=args.seeds[0]
    )

    from torch.utils.data import Subset
    test_ds = Subset(dataset, test_idx)
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn
    )

    # Load ensemble
    ensemble = load_ensemble(args.config, args.seeds, args.ckpt_dir, args.device)

    # Predict
    result = ensemble.predict_loader(test_loader)

    # Calibration
    cal = compute_calibration_metrics(result)
    for k, v in cal.items():
        log.info("  %s = %.4f", k, v)

    # Save
    out_df = result.to_dataframe()
    out_df.to_csv(args.output, index=False)
    log.info("Saved uncertainty predictions â†’ %s", args.output)

    # Reliability diagram
    rel = reliability_diagram_data(result)
    rel.to_csv("reliability_diagram.csv", index=False)
    log.info("Saved reliability diagram â†’ reliability_diagram.csv")

