"""
module10_streamlit.py
=====================
TITLE
Interactive inference and audit application for the kinase-ligand pipeline.

PURPOSE
This module exposes the saved pipeline artefacts through a scientific
Streamlit interface so a new ligand can be scored across the full retained
kinase panel without retraining or rebuilding features.

WHAT IT DOES
- Validates and standardizes a new ligand SMILES.
- Reuses saved ligand/protein features and model checkpoints.
- Scores the ligand across all retained kinase targets.
- Computes uncertainty, model disagreement, and off-target risk.
- Exports tables and audit evidence for the run.

HOW IT WORKS
1. Resolve artefact paths and load cached stores.
2. Standardize and featurize the new ligand.
3. Load retained target metadata and protein features.
4. Run single-model or ensemble inference across the retained panel.
5. Build consensus, selectivity, and exportable result tables.

INPUT CONTRACT
- Saved pipeline artefacts under the configured paths.
- A user-provided SMILES string.

OUTPUT CONTRACT
- Streamlit UI tables, plots, audit panels, and downloadable CSV/JSON outputs.

DEPENDENCIES
- streamlit, pandas, numpy, torch, RDKit
- modules 2, 3, 5, 6, 7, 8, and results_exporter

CRITICAL ASSUMPTIONS
- Artefacts were produced by compatible pipeline modules.
- The retained target list defines the inference universe.

FAILURE MODES
- Invalid SMILES
- Missing checkpoints or feature stores
- No retained targets with available protein features
- Model/config mismatch at inference time

SAFETY CHECKS IMPLEMENTED
- Cache-aware artefact loading
- Explicit invalid-input handling
- Quantitative uncertainty only when available
- Audit views for skipped targets and model errors

HOW TO RUN
- `streamlit run module10_streamlit.py`

HOW IT CONNECTS TO PIPELINE
It is the production-facing inference surface layered directly on top of the
saved outputs from dataset building, feature generation, training, evaluation,
and experiments.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
from rdkit import Chem
from rdkit.Chem import inchi
from torch.utils.data import DataLoader, Dataset

try:
    import altair as alt
except Exception:  # pragma: no cover - optional at runtime
    alt = None

try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
except Exception:  # pragma: no cover - older RDKit fallback
    rdMolStandardize = None

import module8_evaluation as evaluation_module
import results_exporter as exporter_module
from module2_feature_engineering import LigandFeaturizer, load_ligand_feature_store
from module3_protein_features import ProteinFeatures, load_protein_feature_store
from module5_models import ALL_CONFIG_IDS, get_model_config
from module6_training import DEVICE, collate_fn, load_checkpoint, predict
from module7_uncertainty import UncertaintyEnsemble, load_ensemble

log = logging.getLogger("module10_streamlit")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


DEFAULT_PATHS: dict[str, tuple[str, ...]] = {
    "registry": (
        "./registry_clean.parquet",
        "./pipeline_outputs/registry_clean.parquet",
    ),
    "retained_targets": (
        "./retained_targets.csv",
        "./pipeline_outputs/retained_targets.csv",
    ),
    "dataset": (
        "./dataset_clean.parquet",
        "./pipeline_outputs/dataset_clean.parquet",
    ),
    "dataset_summary": (
        "./dataset_summary.csv",
        "./pipeline_outputs/dataset_summary.csv",
    ),
    "protein_store": (
        "./protein_features.pt",
        "./protein_feature_store.pt",
        "./pipeline_outputs/protein_features.pt",
        "./pipeline_outputs/protein_feature_store.pt",
    ),
    "ligand_store": (
        "./ligand_features.pt",
        "./ligand_features",
        "./pipeline_outputs/ligand_features.pt",
        "./pipeline_outputs/ligand_features",
    ),
    "checkpoint_dir": (
        "./checkpoints",
        "./pipeline_outputs/checkpoints",
    ),
    "results_dir": (
        "./results",
        "./pipeline_outputs/results",
    ),
    "config_dir": (
        "./config",
        "./configs",
        "./pipeline_outputs/config",
        "./pipeline_outputs/configs",
    ),
    "artifacts_dir": (
        "./artifacts",
        "./pipeline_outputs/artifacts",
    ),
}


@dataclass(frozen=True)
class ArtifactPaths:
    registry: str
    retained_targets: str
    dataset: str
    dataset_summary: str
    protein_store: str
    ligand_store: str
    checkpoint_dir: str
    results_dir: str
    config_dir: str
    artifacts_dir: str


@dataclass
class ModelBundle:
    config_id: str
    model_cfg: Any
    task_type: str
    seeds: list[int]
    checkpoint_files: list[str]
    parameter_count: Optional[int]
    uncertainty_available: bool
    uncertainty_mode: str
    model: Optional[torch.nn.Module] = None
    ensemble: Optional[UncertaintyEnsemble] = None
    load_error: Optional[str] = None


@dataclass
class StandardizedLigand:
    input_smiles: str
    standardized_smiles: Optional[str]
    canonical_smiles: Optional[str]
    inchikey: Optional[str]
    status: str
    notes: list[str]
    mol: Optional[Chem.Mol]


class KinasePanelInferenceDataset(Dataset):
    """Minimal inference dataset matching module6_training.collate_fn."""

    def __init__(
        self,
        ligand_data: Any,
        target_rows: pd.DataFrame,
        protein_store: dict[str, ProteinFeatures],
    ) -> None:
        self.ligand_data = ligand_data
        self.rows = target_rows.reset_index(drop=True)
        self.protein_store = protein_store

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows.iloc[idx]
        prot = self.protein_store[row["uniprot_id"]]
        return {
            "graph_data": self.ligand_data,
            "physchem": self.ligand_data.physchem,
            "morgan_fp": getattr(self.ligand_data, "morgan_fp", torch.zeros(1024)),
            "esm_pocket": prot.esm_pocket,
            "confidence": prot.confidence,
            "pocket_confidence": torch.tensor(float(getattr(prot, "pocket_confidence", 0.0)), dtype=torch.float),
            "pIC50": torch.tensor(0.0, dtype=torch.float),
            "pIC50_std": torch.tensor(0.5, dtype=torch.float),
            "ic50_nm": torch.tensor(1e3, dtype=torch.float),
            "label": torch.tensor(float("nan"), dtype=torch.float),
            "weight": torch.tensor(1.0, dtype=torch.float),
            "inchikey": getattr(self.ligand_data, "inchikey", ""),
            "uniprot_id": row["uniprot_id"],
            "smiles": getattr(self.ligand_data, "smiles", ""),
            "target_seq": torch.zeros(120, dtype=torch.long),
        }


def _resolve_default_path(key: str) -> str:
    for candidate in DEFAULT_PATHS[key]:
        if Path(candidate).exists():
            return str(Path(candidate))
    return DEFAULT_PATHS[key][0]


def _safe_read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_read_yaml(path: Path) -> Any:
    try:
        import yaml
    except Exception:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _discover_checkpoint_inventory(checkpoint_dir: str) -> dict[str, dict[str, Any]]:
    ckpt_root = Path(checkpoint_dir)
    inventory: dict[str, dict[str, Any]] = {}
    if not ckpt_root.exists():
        return inventory

    pattern = re.compile(r"^(?P<config>.+)_seed(?P<seed>\d+)\.pt$")
    for ckpt in ckpt_root.glob("*.pt"):
        match = pattern.match(ckpt.name)
        if not match:
            continue
        config_id = match.group("config")
        seed = int(match.group("seed"))
        item = inventory.setdefault(config_id, {"seeds": [], "files": []})
        item["seeds"].append(seed)
        item["files"].append(str(ckpt))

    for item in inventory.values():
        item["seeds"] = sorted(set(item["seeds"]))
        item["files"] = sorted(item["files"])
    return inventory


def _best_results_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df
    cols = [c for c in [
        "config_id", "task_type", "family", "spearman_mean", "spearman_std",
        "rmse_mean", "rmse_std", "calibration_mean", "decision_vs_baseline",
    ] if c in results_df.columns]
    return results_df[cols].copy()


def _risk_label_from_delta(delta: float) -> str:
    if delta < 0.5:
        return "strong"
    if delta < 1.0:
        return "moderate"
    if delta >= 1.5:
        return "low"
    return "guarded"


def _confidence_flag(std_value: float, uncertainty_available: bool) -> str:
    if not uncertainty_available or not np.isfinite(std_value):
        return "uncertainty_unavailable"
    if std_value <= 0.30:
        return "tight"
    if std_value <= 0.75:
        return "moderate"
    return "broad"


def _standardize_ligand(smiles: str) -> StandardizedLigand:
    notes: list[str] = []
    raw = (smiles or "").strip()
    if not raw:
        return StandardizedLigand(raw, None, None, None, "empty", ["No SMILES provided."], None)

    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        return StandardizedLigand(raw, None, None, None, "invalid", ["RDKit could not parse the SMILES."], None)

    standardized = mol
    if rdMolStandardize is not None:
        try:
            standardized = rdMolStandardize.Cleanup(standardized)
            standardized = rdMolStandardize.FragmentParent(standardized)
            uncharger = rdMolStandardize.Uncharger()
            standardized = uncharger.uncharge(standardized)
            notes.append("Applied RDKit MolStandardize cleanup, fragment-parent reduction, and uncharging.")
        except Exception as exc:
            notes.append(f"MolStandardize fallback used: {exc}")
            standardized = mol
    else:
        notes.append("MolStandardize unavailable; using RDKit sanitization only.")

    try:
        Chem.SanitizeMol(standardized)
    except Exception as exc:
        return StandardizedLigand(raw, None, None, None, "invalid", [f"Sanitization failed: {exc}"], None)

    canonical = Chem.MolToSmiles(standardized, canonical=True)
    inchikey_value = smiles_to_inchikey(canonical)
    return StandardizedLigand(
        input_smiles=raw,
        standardized_smiles=canonical,
        canonical_smiles=canonical,
        inchikey=inchikey_value,
        status="ok",
        notes=notes or ["SMILES parsed and standardized successfully."],
        mol=standardized,
    )


def canonicalize_smiles(smiles: str) -> str | None:
    standardized = _standardize_ligand(smiles)
    return standardized.canonical_smiles


def smiles_to_inchikey(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return inchi.MolToInchiKey(mol)
    except Exception:
        try:
            inchi_str = inchi.MolToInchi(mol)
            return inchi.InchiToInchiKey(inchi_str) if inchi_str else None
        except Exception:
            return None


@st.cache_data(show_spinner=False)
def load_registry(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    if file_path.suffix.lower() == ".parquet":
        return pd.read_parquet(file_path)
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    raise ValueError(f"Unsupported registry format: {file_path}")


@st.cache_data(show_spinner=False)
def load_retained_targets(path: str, registry_path: str, summary_path: str) -> pd.DataFrame:
    retained = pd.DataFrame()
    path_obj = Path(path)
    if path_obj.exists():
        retained = pd.read_csv(path_obj)

    registry = load_registry(registry_path)
    summary = pd.read_csv(summary_path) if Path(summary_path).exists() else pd.DataFrame()

    if retained.empty and not registry.empty:
        retained = registry.copy()
        if "protein_status" in retained.columns:
            retained = retained[retained["protein_status"] == "retained"].copy()

    if retained.empty:
        return retained

    if not summary.empty:
        merge_cols = [c for c in ["uniprot_id", "target", "pdb_id", "ratio_mode", "n_total", "n_actives", "n_inactives", "pIC50_mean", "pIC50_std", "sources"] if c in summary.columns]
        retained = retained.merge(
            summary[merge_cols].drop_duplicates(subset=["uniprot_id"]),
            on="uniprot_id",
            how="left",
            suffixes=("", "_summary"),
        )

    retained["protein_status"] = retained.get("protein_status", "retained")
    retained["feature_status"] = retained.get("feature_status", "available")
    retained["target"] = retained.get("target", retained["uniprot_id"])
    retained["pdb_id"] = retained.get("pdb_id", "")
    return retained.drop_duplicates(subset=["uniprot_id"]).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_protein_store(path: str) -> dict[str, ProteinFeatures]:
    return load_protein_feature_store(path)


@st.cache_resource(show_spinner=False)
def load_ligand_featurizer() -> LigandFeaturizer:
    return LigandFeaturizer(compute_3d=False, conformer_seed=42)


@st.cache_resource(show_spinner=False)
def load_models(selected_configs: tuple[str, ...], checkpoint_dir: str, device: str) -> dict[str, ModelBundle]:
    inventory = _discover_checkpoint_inventory(checkpoint_dir)
    bundles: dict[str, ModelBundle] = {}
    for config_id in selected_configs:
        try:
            model_cfg = get_model_config(config_id)
        except Exception as exc:
            bundles[config_id] = ModelBundle(
                config_id=config_id,
                model_cfg=None,
                task_type="unknown",
                seeds=[],
                checkpoint_files=[],
                parameter_count=None,
                uncertainty_available=False,
                uncertainty_mode="unavailable",
                load_error=str(exc),
            )
            continue

        available = inventory.get(config_id, {"seeds": [], "files": []})
        seeds = available["seeds"]
        files = available["files"]

        if not seeds:
            bundles[config_id] = ModelBundle(
                config_id=config_id,
                model_cfg=model_cfg,
                task_type=model_cfg.task_type,
                seeds=[],
                checkpoint_files=[],
                parameter_count=None,
                uncertainty_available=False,
                uncertainty_mode="missing_checkpoint",
                load_error=f"No checkpoint files found for config '{config_id}'.",
            )
            continue

        bundle = ModelBundle(
            config_id=config_id,
            model_cfg=model_cfg,
            task_type=model_cfg.task_type,
            seeds=seeds,
            checkpoint_files=files,
            parameter_count=None,
            uncertainty_available=False,
            uncertainty_mode="unavailable",
        )

        try:
            if model_cfg.task_type != "regression":
                model, _ = load_checkpoint(config_id, seeds[0], checkpoint_dir, device)
                bundle.model = model
                bundle.parameter_count = model.count_parameters() if hasattr(model, "count_parameters") else None
                bundle.uncertainty_mode = "classification_not_quantitative"
            elif len(seeds) > 1:
                ensemble = load_ensemble(config_id, seeds, checkpoint_dir, device)
                bundle.ensemble = ensemble
                probe_model = ensemble.models[0]
                bundle.parameter_count = probe_model.count_parameters() if hasattr(probe_model, "count_parameters") else None
                bundle.uncertainty_mode = "ensemble"
                bundle.uncertainty_available = True
            else:
                model, _ = load_checkpoint(config_id, seeds[0], checkpoint_dir, device)
                bundle.model = model
                bundle.parameter_count = model.count_parameters() if hasattr(model, "count_parameters") else None
                bundle.uncertainty_available = bool(getattr(model_cfg, "use_uncertainty", False))
                bundle.uncertainty_mode = "aleatoric_head" if bundle.uncertainty_available else "unavailable"
        except Exception as exc:
            bundle.load_error = str(exc)
        bundles[config_id] = bundle
    return bundles


@st.cache_resource(show_spinner=False)
def _load_ligand_store(path: str) -> Any:
    return load_ligand_feature_store(path)


def _ligand_feature_status(inchikey_value: str, ligand_store: Any) -> tuple[bool, str]:
    try:
        _ = ligand_store[inchikey_value]
        return True, "cache_hit"
    except Exception:
        return False, "computed_in_session"


def _featurize_ligand(canonical_smiles: str, inchikey_value: str, ligand_store_path: str) -> tuple[Any, str]:
    ligand_store = _load_ligand_store(ligand_store_path)
    cached, status = _ligand_feature_status(inchikey_value, ligand_store)
    if cached:
        data = ligand_store[inchikey_value]
        data.inchikey = getattr(data, "inchikey", inchikey_value)
        data.smiles = getattr(data, "smiles", canonical_smiles)
        return data, status

    featurizer = load_ligand_featurizer()
    data = featurizer.featurize(canonical_smiles, inchikey=inchikey_value)
    if data is None:
        raise ValueError("Ligand featurization failed for the standardized molecule.")
    return data, status


def _available_target_panel(
    retained_targets: pd.DataFrame,
    protein_store: dict[str, ProteinFeatures],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if retained_targets.empty:
        return retained_targets.copy(), retained_targets.copy()
    panel = retained_targets.copy()
    panel["protein_available"] = panel["uniprot_id"].isin(set(protein_store.keys()))
    available = panel[panel["protein_available"]].copy().reset_index(drop=True)
    missing = panel[~panel["protein_available"]].copy().reset_index(drop=True)
    return available, missing


def infer_single_model(
    model_bundle: ModelBundle,
    loader: DataLoader,
    target_rows: pd.DataFrame,
    device: str,
) -> pd.DataFrame:
    if model_bundle.task_type != "regression":
        raise ValueError(f"Config '{model_bundle.config_id}' is classification-only and does not output pIC50.")
    if model_bundle.load_error:
        raise RuntimeError(model_bundle.load_error)

    base = target_rows[["target", "uniprot_id", "pdb_id", "protein_status", "feature_status"]].copy().reset_index(drop=True)

    if model_bundle.ensemble is not None:
        unc_result = model_bundle.ensemble.predict_loader(loader)
        base["predicted_pIC50"] = unc_result.pred_mean
        base["uncertainty_std"] = unc_result.pred_std
        base["aleatoric_std"] = np.sqrt(np.clip(unc_result.aleatoric_var, 0.0, None))
        base["epistemic_std"] = np.sqrt(np.clip(unc_result.epistemic_var, 0.0, None))
        base["total_uncertainty_std"] = unc_result.pred_std
        base["lower_95"] = unc_result.lower_95
        base["upper_95"] = unc_result.upper_95
        base["uncertainty_type"] = "ensemble_decomposed"
        base["model_disagreement"] = np.sqrt(np.clip(unc_result.epistemic_var, 0.0, None))
    else:
        if model_bundle.model is None:
            raise RuntimeError(f"Model object missing for '{model_bundle.config_id}'.")
        preds = predict(model_bundle.model, loader, device)
        mu = np.asarray(preds["mu"], dtype=float)
        var = np.asarray(preds.get("var", np.full_like(mu, np.nan)), dtype=float)
        std = np.sqrt(np.clip(var, 0.0, None)) if np.isfinite(var).any() else np.full_like(mu, np.nan)
        base["predicted_pIC50"] = mu
        base["uncertainty_std"] = std if model_bundle.uncertainty_available else np.nan
        base["aleatoric_std"] = std if model_bundle.uncertainty_available else np.nan
        base["epistemic_std"] = 0.0 if model_bundle.uncertainty_available else np.nan
        base["total_uncertainty_std"] = std if model_bundle.uncertainty_available else np.nan
        base["lower_95"] = mu - 1.96 * std if model_bundle.uncertainty_available else np.nan
        base["upper_95"] = mu + 1.96 * std if model_bundle.uncertainty_available else np.nan
        base["uncertainty_type"] = "aleatoric_head" if model_bundle.uncertainty_available else "unavailable"
        base["model_disagreement"] = np.nan

    base["model_name"] = model_bundle.config_id
    base["predicted_IC50_nM"] = compute_ic50_from_pic50(base["predicted_pIC50"].to_numpy())
    base["rank"] = base["predicted_pIC50"].rank(method="dense", ascending=False).astype(int)
    base["confidence_flag"] = [
        _confidence_flag(float(std_value) if pd.notna(std_value) else float("nan"), model_bundle.uncertainty_available)
        for std_value in base["uncertainty_std"]
    ]
    return base.sort_values("predicted_pIC50", ascending=False).reset_index(drop=True)


def infer_across_kinases(
    ligand_data: Any,
    retained_targets: pd.DataFrame,
    protein_store: dict[str, ProteinFeatures],
    models: dict[str, ModelBundle],
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    available_targets, missing_targets = _available_target_panel(retained_targets, protein_store)
    if available_targets.empty:
        raise ValueError("No retained targets have matching protein features in the store.")

    dataset = KinasePanelInferenceDataset(ligand_data, available_targets, protein_store)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model_frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    for config_id, bundle in models.items():
        if bundle.task_type != "regression":
            errors[config_id] = "Classification checkpoint loaded; quantitative pIC50 inference is skipped."
            continue
        try:
            frame = infer_single_model(bundle, loader, available_targets, device)
            model_frames.append(frame)
        except Exception as exc:
            errors[config_id] = str(exc)

    prediction_df = build_prediction_dataframe(model_frames)
    consensus_df = pd.DataFrame()
    if not prediction_df.empty:
        consensus_df = (
            prediction_df.groupby(["target", "uniprot_id", "pdb_id", "protein_status", "feature_status"], as_index=False)
            .agg(
                consensus_pIC50=("predicted_pIC50", "mean"),
                consensus_IC50_nM=("predicted_IC50_nM", "mean"),
                consensus_uncertainty_std=("uncertainty_std", "mean"),
                mean_aleatoric_std=("aleatoric_std", "mean"),
                mean_epistemic_std=("epistemic_std", "mean"),
                lower_95=("lower_95", "mean"),
                upper_95=("upper_95", "mean"),
                model_count=("model_name", "nunique"),
                model_disagreement=("predicted_pIC50", "std"),
            )
        )
        consensus_df["model_disagreement"] = consensus_df["model_disagreement"].fillna(0.0)
        consensus_df["rank"] = consensus_df["consensus_pIC50"].rank(method="dense", ascending=False).astype(int)
        consensus_df["confidence_flag"] = [
            _confidence_flag(float(v) if pd.notna(v) else float("nan"), pd.notna(v))
            for v in consensus_df["consensus_uncertainty_std"]
        ]
        consensus_df = consensus_df.sort_values("consensus_pIC50", ascending=False).reset_index(drop=True)

    return {
        "prediction_df": prediction_df,
        "consensus_df": consensus_df,
        "available_targets": available_targets,
        "missing_targets": missing_targets,
        "errors": errors,
    }


def compute_ic50_from_pic50(pic50: Any) -> Any:
    values = np.asarray(pic50, dtype=float)
    ic50 = np.power(10.0, 9.0 - values)
    if np.isscalar(pic50):
        return float(ic50)
    return ic50


def compute_selectivity_table(
    prediction_frame: pd.DataFrame,
    reference_target: str,
    score_column: str = "consensus_pIC50",
) -> pd.DataFrame:
    if prediction_frame.empty or score_column not in prediction_frame.columns:
        return pd.DataFrame()

    ref_mask = (prediction_frame["uniprot_id"] == reference_target) | (prediction_frame["target"] == reference_target)
    if not ref_mask.any():
        return pd.DataFrame()

    ref_row = prediction_frame.loc[ref_mask].iloc[0]
    ref_score = float(ref_row[score_column])

    selectivity = prediction_frame.copy()
    selectivity["reference_target"] = ref_row["target"]
    selectivity["reference_uniprot_id"] = ref_row["uniprot_id"]
    selectivity["reference_pIC50"] = ref_score
    selectivity["delta_pIC50_vs_reference"] = ref_score - selectivity[score_column].astype(float)
    selectivity["off_target_risk_label"] = selectivity["delta_pIC50_vs_reference"].map(_risk_label_from_delta)
    selectivity["risk_score_component"] = 1.0 / (1.0 + np.exp(selectivity["delta_pIC50_vs_reference"].astype(float)))
    selectivity["is_reference"] = selectivity["uniprot_id"] == ref_row["uniprot_id"]
    selectivity = selectivity.sort_values(
        by=["is_reference", "delta_pIC50_vs_reference"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return selectivity


def compute_off_target_risk(selectivity_df: pd.DataFrame) -> dict[str, Any]:
    if selectivity_df.empty:
        return {
            "risk_score": np.nan,
            "high_risk_count": 0,
            "moderate_risk_count": 0,
            "low_risk_count": 0,
            "fraction_within_1_log": np.nan,
            "summary": "No selectivity table available.",
            "high_risk_targets": [],
            "moderate_risk_targets": [],
        }

    off_target_df = selectivity_df[~selectivity_df["is_reference"]].copy()
    if off_target_df.empty:
        return {
            "risk_score": 0.0,
            "high_risk_count": 0,
            "moderate_risk_count": 0,
            "low_risk_count": 0,
            "fraction_within_1_log": 0.0,
            "summary": "Only the reference target was available, so off-target risk is effectively zero.",
            "high_risk_targets": [],
            "moderate_risk_targets": [],
        }

    high = off_target_df[off_target_df["off_target_risk_label"] == "strong"]
    moderate = off_target_df[off_target_df["off_target_risk_label"] == "moderate"]
    low = off_target_df[off_target_df["off_target_risk_label"] == "low"]
    risk_score = float(off_target_df["risk_score_component"].mean())
    frac_within_1 = float((off_target_df["delta_pIC50_vs_reference"] < 1.0).mean())
    summary = (
        f"Weighted off-target risk score = {risk_score:.3f} using mean(sigmoid(-Î”pIC50)). "
        f"{len(high)} kinases fall in the strong-risk zone (Î”pIC50 < 0.5), "
        f"{len(moderate)} are moderate (Î”pIC50 < 1.0), and "
        f"{len(low)} are low-risk (Î”pIC50 >= 1.5)."
    )
    return {
        "risk_score": risk_score,
        "high_risk_count": int(len(high)),
        "moderate_risk_count": int(len(moderate)),
        "low_risk_count": int(len(low)),
        "fraction_within_1_log": frac_within_1,
        "summary": summary,
        "high_risk_targets": high["target"].tolist(),
        "moderate_risk_targets": moderate["target"].tolist(),
    }


def build_prediction_dataframe(model_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not model_frames:
        return pd.DataFrame(columns=[
            "target", "uniprot_id", "pdb_id", "model_name", "predicted_pIC50",
            "predicted_IC50_nM", "uncertainty_std", "uncertainty_type", "rank",
            "confidence_flag", "protein_status", "feature_status",
        ])
    frame = pd.concat(model_frames, ignore_index=True)
    ordered_cols = [
        "target", "uniprot_id", "pdb_id", "model_name", "predicted_pIC50",
        "predicted_IC50_nM", "uncertainty_std", "uncertainty_type",
        "aleatoric_std", "epistemic_std", "total_uncertainty_std",
        "model_disagreement", "lower_95", "upper_95", "rank", "confidence_flag",
        "protein_status", "feature_status",
    ]
    return frame[[c for c in ordered_cols if c in frame.columns]].copy()


def build_off_target_dataframe(selectivity_df: pd.DataFrame) -> pd.DataFrame:
    if selectivity_df.empty:
        return pd.DataFrame(columns=[
            "target", "uniprot_id", "predicted_pIC50", "predicted_IC50_nM",
            "delta_pIC50_vs_reference", "off_target_risk_label", "risk_score_component",
        ])

    out = selectivity_df.copy()
    if "consensus_pIC50" in out.columns:
        out["predicted_pIC50"] = out["consensus_pIC50"]
    if "consensus_IC50_nM" in out.columns:
        out["predicted_IC50_nM"] = out["consensus_IC50_nM"]
    keep = [
        "target", "uniprot_id", "predicted_pIC50", "predicted_IC50_nM",
        "delta_pIC50_vs_reference", "off_target_risk_label", "risk_score_component",
        "reference_target", "reference_uniprot_id", "is_reference",
    ]
    return out[[c for c in keep if c in out.columns]].copy()


@st.cache_data(show_spinner=False)
def _load_results_tables(results_dir: str) -> dict[str, pd.DataFrame]:
    root = Path(results_dir)
    outputs: dict[str, pd.DataFrame] = {}
    for name in ["results.csv", "per_seed_results.csv", "per_kinase_results.csv", "uncertainty_predictions.csv", "reliability_diagram.csv"]:
        path = root / name
        outputs[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return outputs


@st.cache_data(show_spinner=False)
def _load_auxiliary_metadata(results_dir: str, config_dir: str, artifacts_dir: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    results_root = Path(results_dir)
    for name in ["training_summary.json", "calibration_metrics.json", "experiment_summary.txt"]:
        path = results_root / name
        if not path.exists():
            continue
        if path.suffix == ".json":
            payload[name] = _safe_read_json(path)
        else:
            payload[name] = path.read_text(encoding="utf-8")

    config_files: list[dict[str, Any]] = []
    for root_str in [config_dir, artifacts_dir, results_dir]:
        root = Path(root_str)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
                continue
            parsed = None
            if path.suffix.lower() == ".json":
                parsed = _safe_read_json(path)
            else:
                parsed = _safe_read_yaml(path)
            config_files.append({"path": str(path), "content": parsed})
    payload["saved_config_files"] = config_files[:20]
    return payload


@st.cache_data(show_spinner=True)
def _run_scoring_cached(
    smiles: str,
    selected_configs: tuple[str, ...],
    selected_target: str,
    paths_payload: dict[str, str],
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    standardized = _standardize_ligand(smiles)
    if standardized.status != "ok" or not standardized.canonical_smiles or not standardized.inchikey:
        raise ValueError("; ".join(standardized.notes) or "Ligand standardization failed.")

    retained_targets = load_retained_targets(
        paths_payload["retained_targets"],
        paths_payload["registry"],
        paths_payload["dataset_summary"],
    )
    if retained_targets.empty:
        raise ValueError("Retained target set is empty. Check retained_targets.csv or registry_clean.parquet.")

    protein_store = load_protein_store(paths_payload["protein_store"])
    ligand_data, feature_source = _featurize_ligand(
        standardized.canonical_smiles,
        standardized.inchikey,
        paths_payload["ligand_store"],
    )
    models = load_models(selected_configs, paths_payload["checkpoint_dir"], device)
    inference = infer_across_kinases(
        ligand_data=ligand_data,
        retained_targets=retained_targets,
        protein_store=protein_store,
        models=models,
        batch_size=batch_size,
        device=device,
    )

    consensus_df = inference["consensus_df"]
    selectivity_df = compute_selectivity_table(
        consensus_df if not consensus_df.empty else inference["prediction_df"],
        reference_target=selected_target,
        score_column="consensus_pIC50" if not consensus_df.empty else "predicted_pIC50",
    )
    off_target_df = build_off_target_dataframe(selectivity_df)
    off_target_risk = compute_off_target_risk(selectivity_df)

    return {
        "standardized": {
            "input_smiles": standardized.input_smiles,
            "canonical_smiles": standardized.canonical_smiles,
            "inchikey": standardized.inchikey,
            "status": standardized.status,
            "notes": standardized.notes,
            "feature_source": feature_source,
        },
        "prediction_df": inference["prediction_df"],
        "consensus_df": consensus_df,
        "available_targets": inference["available_targets"],
        "missing_targets": inference["missing_targets"],
        "errors": inference["errors"],
        "off_target_df": off_target_df,
        "off_target_risk": off_target_risk,
        "selected_target": selected_target,
    }


def _render_ranked_chart(df: pd.DataFrame, top_n: int, score_column: str, uncertainty_column: Optional[str]) -> None:
    chart_df = df.nsmallest(top_n, "rank").copy() if "rank" in df.columns else df.head(top_n).copy()
    if chart_df.empty:
        st.info("No ranked predictions available yet.")
        return
    chart_df["label"] = chart_df["target"] + " (" + chart_df["uniprot_id"] + ")"

    if alt is None:
        st.bar_chart(chart_df.set_index("label")[score_column])
        return

    bars = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X(score_column, title=score_column),
        y=alt.Y("label:N", sort="-x", title="Target"),
        tooltip=["target", "uniprot_id", "pdb_id", score_column],
    )
    if uncertainty_column and uncertainty_column in chart_df.columns and chart_df[uncertainty_column].notna().any():
        err_df = chart_df.copy()
        err_df["err_low"] = err_df[score_column] - err_df[uncertainty_column]
        err_df["err_high"] = err_df[score_column] + err_df[uncertainty_column]
        errors = alt.Chart(err_df).mark_errorbar().encode(
            x="err_low:Q",
            x2="err_high:Q",
            y=alt.Y("label:N", sort="-x"),
        )
        st.altair_chart((bars + errors).properties(height=max(320, top_n * 18)), use_container_width=True)
    else:
        st.altair_chart(bars.properties(height=max(320, top_n * 18)), use_container_width=True)


def _render_heatmap(prediction_df: pd.DataFrame, top_n: int) -> None:
    if alt is None or prediction_df.empty or prediction_df["model_name"].nunique() < 2:
        return
    ranks = prediction_df.groupby("uniprot_id")["predicted_pIC50"].mean().reset_index()
    top_targets = ranks.sort_values("predicted_pIC50", ascending=False).head(top_n)["uniprot_id"]
    heat_df = prediction_df[prediction_df["uniprot_id"].isin(top_targets)].copy()
    heat_df["label"] = heat_df["target"] + " (" + heat_df["uniprot_id"] + ")"
    chart = alt.Chart(heat_df).mark_rect().encode(
        x=alt.X("model_name:N", title="Model"),
        y=alt.Y("label:N", sort="-x", title="Target"),
        color=alt.Color("predicted_pIC50:Q", title="Predicted pIC50"),
        tooltip=["target", "uniprot_id", "model_name", "predicted_pIC50"],
    )
    st.altair_chart(chart.properties(height=max(320, top_n * 18)), use_container_width=True)


def _render_delta_plot(off_target_df: pd.DataFrame, top_n: int) -> None:
    if off_target_df.empty:
        st.info("Select a target and run inference to compute off-target deltas.")
        return
    if "is_reference" in off_target_df.columns:
        plot_df = off_target_df[~off_target_df["is_reference"]].copy()
    else:
        plot_df = off_target_df.copy()
    plot_df = plot_df.sort_values("delta_pIC50_vs_reference")
    plot_df = plot_df.head(top_n)
    if alt is None:
        st.bar_chart(plot_df.set_index("target")["delta_pIC50_vs_reference"])
        return
    chart = alt.Chart(plot_df).mark_bar().encode(
        x=alt.X("delta_pIC50_vs_reference:Q", title="Î”pIC50 vs reference"),
        y=alt.Y("target:N", sort="x"),
        color=alt.Color("off_target_risk_label:N", title="Risk"),
        tooltip=["target", "uniprot_id", "delta_pIC50_vs_reference", "off_target_risk_label"],
    )
    st.altair_chart(chart.properties(height=max(320, top_n * 18)), use_container_width=True)


def _render_download_button(label: str, df: pd.DataFrame, filename: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv_bytes, file_name=filename, mime="text/csv")


def render_ui() -> None:
    st.set_page_config(
        page_title="Kinase-Ligand Inference Panel",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Kinase-Ligand Full Panel Inference")
    st.caption(
        "Inference-only Streamlit front-end over retained kinase targets, stored protein features, "
        "saved model checkpoints, and exported evaluation artefacts."
    )

    for key, default in {
        "selected_ligand_input": "",
        "selected_target_kinase": "",
        "selected_model_configs": [],
        "last_predictions": None,
    }.items():
        st.session_state.setdefault(key, default)

    with st.sidebar:
        st.header("Artifacts")
        path_values = {
            "registry": st.text_input("Registry", value=_resolve_default_path("registry")),
            "retained_targets": st.text_input("Retained targets", value=_resolve_default_path("retained_targets")),
            "dataset": st.text_input("Dataset", value=_resolve_default_path("dataset")),
            "dataset_summary": st.text_input("Dataset summary", value=_resolve_default_path("dataset_summary")),
            "protein_store": st.text_input("Protein store", value=_resolve_default_path("protein_store")),
            "ligand_store": st.text_input("Ligand store", value=_resolve_default_path("ligand_store")),
            "checkpoint_dir": st.text_input("Checkpoint dir", value=_resolve_default_path("checkpoint_dir")),
            "results_dir": st.text_input("Results dir", value=_resolve_default_path("results_dir")),
            "config_dir": st.text_input("Config dir", value=_resolve_default_path("config_dir")),
            "artifacts_dir": st.text_input("Artifacts dir", value=_resolve_default_path("artifacts_dir")),
        }
        device_options = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        chosen_device = st.selectbox("Inference device", device_options, index=device_options.index("cuda") if "cuda" in device_options and str(DEVICE).startswith("cuda") else 0)
        batch_size = st.slider("Inference batch size", min_value=8, max_value=256, value=64 if chosen_device == "cuda" else 32, step=8)
        top_n = st.slider("Top-N for plots", min_value=10, max_value=100, value=30, step=5)

    retained_targets = load_retained_targets(path_values["retained_targets"], path_values["registry"], path_values["dataset_summary"])
    results_tables = _load_results_tables(path_values["results_dir"])
    aux_meta = _load_auxiliary_metadata(path_values["results_dir"], path_values["config_dir"], path_values["artifacts_dir"])
    checkpoint_inventory = _discover_checkpoint_inventory(path_values["checkpoint_dir"])

    available_configs = sorted(checkpoint_inventory.keys())
    regression_defaults = [cid for cid in available_configs if cid in ALL_CONFIG_IDS and get_model_config(cid).task_type == "regression"][:3]
    if not st.session_state["selected_model_configs"]:
        st.session_state["selected_model_configs"] = regression_defaults
    if retained_targets is not None and not retained_targets.empty and not st.session_state["selected_target_kinase"]:
        st.session_state["selected_target_kinase"] = retained_targets.iloc[0]["uniprot_id"]

    control_col1, control_col2 = st.columns([2, 2])
    with control_col1:
        st.session_state["selected_model_configs"] = st.multiselect(
            "Model configs",
            options=available_configs,
            default=st.session_state["selected_model_configs"],
            help="Saved checkpoints discovered in the checkpoint directory.",
        )
    with control_col2:
        target_options = retained_targets["uniprot_id"].tolist() if not retained_targets.empty else []
        st.session_state["selected_target_kinase"] = st.selectbox(
            "Reference target for selectivity",
            options=target_options,
            index=target_options.index(st.session_state["selected_target_kinase"]) if target_options and st.session_state["selected_target_kinase"] in target_options else 0,
            format_func=lambda uid: (
                f"{retained_targets.loc[retained_targets['uniprot_id'] == uid, 'target'].iloc[0]} [{uid}]"
                if uid in set(retained_targets["uniprot_id"]) else uid
            ),
        ) if target_options else ""

    tabs = st.tabs([
        "Input and Validation",
        "Model Selection",
        "Target Selection",
        "Full Kinase Scoring Panel",
        "Selectivity and Off-Target Analysis",
        "Evidence and Audit",
        "Export",
    ])

    with tabs[0]:
        with st.form("ligand_form"):
            ligand_input = st.text_area(
                "New ligand SMILES",
                value=st.session_state["selected_ligand_input"],
                height=120,
                placeholder="Enter a SMILES string to score across the full retained kinase panel.",
            )
            submitted = st.form_submit_button("Run full kinase scoring panel")

        preview = _standardize_ligand(ligand_input)
        if preview.status == "ok":
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Canonical SMILES", preview.canonical_smiles or "NA")
            col_b.metric("InChIKey", preview.inchikey or "NA")
            col_c.metric("Standardization status", preview.status)
            st.write("Standardization notes:")
            for note in preview.notes:
                st.write(f"- {note}")
        elif ligand_input.strip():
            st.error("; ".join(preview.notes))

        if submitted:
            st.session_state["selected_ligand_input"] = ligand_input
            if not st.session_state["selected_model_configs"]:
                st.error("Select at least one model configuration with saved checkpoints.")
            else:
                try:
                    with st.spinner("Running inference across all retained kinase targets..."):
                        result = _run_scoring_cached(
                            smiles=ligand_input,
                            selected_configs=tuple(st.session_state["selected_model_configs"]),
                            selected_target=st.session_state["selected_target_kinase"],
                            paths_payload=path_values,
                            batch_size=batch_size,
                            device=chosen_device,
                        )
                    st.session_state["last_predictions"] = result
                    st.success("Inference completed.")
                except Exception as exc:
                    log.exception("Inference failed")
                    st.error(f"Inference failed: {exc}")

    with tabs[1]:
        model_bundles = load_models(tuple(st.session_state["selected_model_configs"]), path_values["checkpoint_dir"], chosen_device) if st.session_state["selected_model_configs"] else {}
        model_rows = []
        for config_id in st.session_state["selected_model_configs"]:
            bundle = model_bundles.get(config_id)
            if bundle is None:
                continue
            model_rows.append({
                "config_id": config_id,
                "task_type": bundle.task_type,
                "checkpoint_count": len(bundle.checkpoint_files),
                "seeds": ",".join(map(str, bundle.seeds)),
                "parameter_count": bundle.parameter_count,
                "uncertainty_mode": bundle.uncertainty_mode,
                "uncertainty_available": bundle.uncertainty_available,
                "load_error": bundle.load_error or "",
            })
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)
        if not results_tables["results.csv"].empty:
            st.subheader("Saved evaluation summary")
            st.dataframe(_best_results_metrics(results_tables["results.csv"]), use_container_width=True, hide_index=True)

    with tabs[2]:
        if retained_targets.empty:
            st.warning("No retained target metadata found.")
        else:
            target_row = retained_targets[retained_targets["uniprot_id"] == st.session_state["selected_target_kinase"]]
            if target_row.empty:
                st.info("Select a retained target to inspect metadata.")
            else:
                row = target_row.iloc[0]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Target", str(row.get("target", "")))
                col2.metric("UniProt", str(row.get("uniprot_id", "")))
                col3.metric("PDB", str(row.get("pdb_id", "")))
                col4.metric("Protein features", "available" if Path(path_values["protein_store"]).exists() else "missing")
                compact_cols = [c for c in ["n_total", "n_actives", "n_inactives", "ratio_mode", "pIC50_mean", "pIC50_std", "sources"] if c in target_row.columns]
                if compact_cols:
                    st.dataframe(target_row[compact_cols], use_container_width=True, hide_index=True)

    result = st.session_state.get("last_predictions")

    with tabs[3]:
        if not result:
            st.info("Run inference to populate the full kinase scoring panel.")
        else:
            consensus_df = result["consensus_df"]
            prediction_df = result["prediction_df"]
            if not consensus_df.empty:
                top_cols = [c for c in [
                    "rank", "target", "uniprot_id", "pdb_id", "consensus_pIC50", "consensus_IC50_nM",
                    "consensus_uncertainty_std", "model_disagreement", "confidence_flag", "protein_status", "feature_status",
                ] if c in consensus_df.columns]
                st.subheader("Consensus scoring panel")
                st.dataframe(consensus_df[top_cols], use_container_width=True, hide_index=True)
                _render_ranked_chart(consensus_df, top_n=top_n, score_column="consensus_pIC50", uncertainty_column="consensus_uncertainty_std")
            st.subheader("Per-model predictions")
            st.dataframe(prediction_df, use_container_width=True, hide_index=True)
            _render_heatmap(prediction_df, top_n=min(top_n, 40))
            if result["errors"]:
                st.warning("Some selected models were skipped.")
                st.json(result["errors"])

    with tabs[4]:
        if not result:
            st.info("Run inference to compute selectivity and off-target analysis.")
        else:
            off_target_df = result["off_target_df"]
            risk = result["off_target_risk"]
            if not off_target_df.empty and "is_reference" in off_target_df.columns:
                ref_row = off_target_df[off_target_df["is_reference"]]
            else:
                ref_row = pd.DataFrame()
            if not ref_row.empty:
                selected = ref_row.iloc[0]
                col1, col2, col3 = st.columns(3)
                col1.metric("Selected target", f"{selected['target']} [{selected['uniprot_id']}]")
                col2.metric("Predicted pIC50", f"{selected['predicted_pIC50']:.3f}")
                col3.metric("Predicted IC50 (nM)", f"{selected['predicted_IC50_nM']:.2f}")
            score_value = 0.0 if not np.isfinite(risk["risk_score"]) else min(max(risk["risk_score"], 0.0), 1.0)
            st.metric("Off-target risk score", f"{risk['risk_score']:.3f}" if np.isfinite(risk["risk_score"]) else "NA")
            st.progress(score_value)
            st.write(risk["summary"])
            if risk["high_risk_targets"]:
                st.error("High-risk kinases: " + ", ".join(risk["high_risk_targets"]))
            if risk["moderate_risk_targets"]:
                st.warning("Moderate-risk kinases: " + ", ".join(risk["moderate_risk_targets"]))
            st.dataframe(off_target_df, use_container_width=True, hide_index=True)
            _render_delta_plot(off_target_df, top_n=min(top_n, 40))

    with tabs[5]:
        standardized = result["standardized"] if result else None
        st.subheader("Ligand evidence")
        if standardized:
            st.json(standardized)
        else:
            st.info("Ligand audit details will appear after inference.")

        st.subheader("Target and checkpoint evidence")
        evidence = {
            "retained_target_source": path_values["retained_targets"],
            "registry_source": path_values["registry"],
            "protein_store_source": path_values["protein_store"],
            "ligand_store_source": path_values["ligand_store"],
            "checkpoint_dir": path_values["checkpoint_dir"],
            "results_dir": path_values["results_dir"],
            "module8_evaluation_loaded": evaluation_module.__name__,
            "results_exporter_loaded": exporter_module.__name__,
        }
        st.json(evidence)

        if result:
            st.write(f"Included targets with protein features: {len(result['available_targets'])}")
            if not result["missing_targets"].empty:
                st.write("Targets skipped because protein features were unavailable:")
                st.dataframe(result["missing_targets"], use_container_width=True, hide_index=True)

        dataset_summary = pd.read_csv(path_values["dataset_summary"]) if Path(path_values["dataset_summary"]).exists() else pd.DataFrame()
        if not dataset_summary.empty:
            ratio_modes = sorted(dataset_summary["ratio_mode"].dropna().unique().tolist()) if "ratio_mode" in dataset_summary.columns else []
            st.write("Dataset summary and ratio mode")
            st.write(f"Ratio mode(s): {', '.join(ratio_modes) if ratio_modes else 'NA'}")
            st.dataframe(dataset_summary.head(50), use_container_width=True, hide_index=True)

        if aux_meta.get("training_summary.json"):
            st.subheader("Training summary")
            st.json(aux_meta["training_summary.json"])
        if aux_meta.get("calibration_metrics.json"):
            st.subheader("Calibration metrics")
            st.json(aux_meta["calibration_metrics.json"])
        if aux_meta.get("saved_config_files"):
            st.subheader("Saved config files")
            st.dataframe(pd.DataFrame(aux_meta["saved_config_files"]), use_container_width=True, hide_index=True)

    with tabs[6]:
        if not result:
            st.info("Run inference to enable export.")
        else:
            prediction_df = result["prediction_df"]
            consensus_df = result["consensus_df"]
            off_target_df = result["off_target_df"]
            report_payload = {
                "ligand": result["standardized"],
                "selected_target": result["selected_target"],
                "off_target_risk": result["off_target_risk"],
                "model_errors": result["errors"],
                "artifact_paths": path_values,
            }
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                _render_download_button("Export prediction table", prediction_df, "kinase_predictions.csv")
            with col2:
                _render_download_button("Export off-target table", off_target_df, "off_target_analysis.csv")
            with col3:
                _render_download_button("Export consensus table", consensus_df, "consensus_predictions.csv")
            with col4:
                st.download_button(
                    label="Export run report",
                    data=json.dumps(report_payload, indent=2).encode("utf-8"),
                    file_name="streamlit_inference_report.json",
                    mime="application/json",
                )


if __name__ == "__main__":
    render_ui()

