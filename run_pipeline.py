"""
run_pipeline.py
===============
TITLE
Crash-resistant master entrypoint for the kinase-ligand pipeline.

PURPOSE
This file is the only command-line entrypoint a team should need. It validates
the environment, checks artefacts, rebuilds missing stages when safe, aligns
features before training, and leaves a diagnostic trail when anything goes
wrong.

WHAT IT DOES
1. Validates Python/runtime dependencies and hardware visibility.
2. Validates required files and resolves canonical artefact paths.
3. Diagnoses the current repository state before execution.
4. Rebuilds or reuses dataset, ligand features, protein features, checkpoints,
   evaluation tables, and experiment exports according to cache policy.
5. Normalizes column names globally so downstream code does not crash on common
   schema variants such as `pic50`, `uniprot`, or `canonical_smiles`.
6. Aligns the dataset against both ligand and protein feature stores before
   training or evaluation.
7. Stops safely when the configured time budget is exhausted.
8. Optionally launches the Streamlit app after the pipeline is healthy.

HOW IT WORKS
1. Build a `PipelinePaths` object rooted at `--work-dir`.
2. Run preflight environment and file validation.
3. Print and persist a diagnostics report.
4. Reuse protein features if present; otherwise defer building until retained
   targets are available from the dataset stage.
5. Build or load the dataset and normalize its schema.
6. Build or load ligand features.
7. Build or load protein features for retained targets.
8. Intersect dataset rows with available ligand/protein features.
9. Train requested model configs unless checkpoints already exist and cache is
   enabled or `--skip_training` is passed.
10. Validate checkpoints, run evaluation, run time-bounded experiments, and
    optionally launch Streamlit.

INPUT CONTRACT
- Input registry workbook: Excel file with at least `target`, `uniprot_id`,
  and `pdb_id`, unless cached dataset artefacts already exist.
- Clean dataset: parquet/CSV containing normalized columns for ligand-target
  records. This entrypoint can rebuild it if missing.
- Feature stores: ligand and protein stores may be absent; they will be built
  if prerequisites exist.
- Checkpoints: optional; if missing and training is enabled they will be built.

OUTPUT CONTRACT
- Normalized and aligned dataset artefacts under `--work-dir`.
- Ligand and protein feature stores.
- Checkpoints under `checkpoints/`.
- Evaluation tables under `results/`.
- Diagnostics JSON and alignment reports for auditability.

DEPENDENCIES
- module1_dataset_builder.py
- module2_feature_engineering.py
- module3_protein_features.py
- module5_models.py
- module6_training.py
- module8_evaluation.py
- module9_experiments.py
- module10_streamlit.py
- results_exporter.py

CRITICAL ASSUMPTIONS
- The repository is writable inside `--work-dir`.
- RDKit, PyTorch, and parquet support are installed if their stages are used.
- Existing cached artefacts may be stale, so this file validates them before
  trusting them.

FAILURE MODES
- Missing Excel registry when no cached dataset exists.
- Empty dataset after build or after feature alignment.
- Missing required columns after normalization.
- Missing checkpoints after training.
- Time budget exhaustion before the next expensive stage.

SAFETY CHECKS IMPLEMENTED
- Environment validation and GPU visibility report.
- Required-file checks with cache-aware fallbacks.
- Global column normalization.
- Dataset schema validation.
- Feature alignment and minimum-row threshold checks.
- Checkpoint existence validation after training.
- Stage-by-stage time-budget enforcement.
- Persistent diagnostics and alignment reports.

HOW TO RUN
- Full pipeline:
  `python run_pipeline.py --ratio_mode 1:2 --time_limit_hours 24 --use_cache`
- Dataset only:
  `python run_pipeline.py --only_stage dataset`
- Features only:
  `python run_pipeline.py --only_stage features --use_cache`
- Training only:
  `python run_pipeline.py --only_stage train --use_cache`
- Evaluation only:
  `python run_pipeline.py --only_stage eval --use_cache`
- Streamlit:
  `streamlit run module10_streamlit.py`

HOW IT CONNECTS TO PIPELINE
It orchestrates modules 1-10 in a safe order, normalizes their artefacts, and
exposes one reliable operational path for both batch runs and app launch.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from module1_dataset_builder import build_dataset
from module2_feature_engineering import build_ligand_feature_store, load_ligand_feature_store
from module3_protein_features import build_feature_store, load_protein_feature_store
from module5_models import ALL_CONFIG_IDS, get_model_config
from module6_training import DEVICE, KinaseLigandDataset, TrainConfig, train_all_configs
from module8_evaluation import build_results_csv, run_full_evaluation, save_per_kinase_results
from module9_experiments import ExperimentConfig, PRIORITY_CONFIGS, run_time_bounded_ablation

log = logging.getLogger("run_pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

REQUIRED_DATASET_COLUMNS = {
    "target",
    "uniprot_id",
    "pdb_id",
    "smiles",
    "inchikey",
    "pIC50",
}
MIN_ALIGNED_ROWS = 25

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "pIC50": ("pIC50", "pic50", "pIC50_median", "pic50_median"),
    "ic50_nm_median": ("ic50_nm_median", "IC50", "ic50_nm", "ic50", "IC50_nM"),
    "uniprot_id": ("uniprot_id", "uniprot", "uniprot_accession", "accession"),
    "smiles": ("smiles", "canonical_smiles", "SMILES"),
    "inchikey": ("inchikey", "InChIKey", "inchi_key"),
    "target": ("target", "target_name", "gene_symbol", "kinase"),
    "pdb_id": ("pdb_id", "pdb", "PDB_ID"),
}


class PipelineStop(RuntimeError):
    """Raised for safe, user-facing pipeline termination."""


@dataclass
class PipelinePaths:
    work_dir: Path
    excel: Path
    dataset: Path
    retained_targets: Path
    registry: Path
    dataset_summary: Path
    ligand_store: Path
    protein_store: Path
    checkpoint_dir: Path
    results_dir: Path
    diagnostics_path: Path
    alignment_report_path: Path
    run_state_path: Path


def build_paths(work_dir: str, excel_path: str) -> PipelinePaths:
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    return PipelinePaths(
        work_dir=root,
        excel=Path(excel_path),
        dataset=root / "dataset_clean.parquet",
        retained_targets=root / "retained_targets.csv",
        registry=root / "registry_clean.parquet",
        dataset_summary=root / "dataset_summary.csv",
        ligand_store=root / "ligand_features",
        protein_store=root / "protein_feature_store.pt",
        checkpoint_dir=root / "checkpoints",
        results_dir=root / "results",
        diagnostics_path=root / "pipeline_diagnostics.json",
        alignment_report_path=root / "feature_alignment_report.json",
        run_state_path=root / "pipeline_run_state.json",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _time_exceeded(start_time: float, time_limit_hours: float) -> bool:
    return (time.time() - start_time) >= time_limit_hours * 3600


def _assert_time_budget(stage_name: str, start_time: float, time_limit_hours: float, paths: PipelinePaths) -> None:
    if _time_exceeded(start_time, time_limit_hours):
        state = {
            "status": "stopped_time_limit",
            "stage": stage_name,
            "elapsed_seconds": time.time() - start_time,
            "time_limit_hours": time_limit_hours,
        }
        _write_json(paths.run_state_path, state)
        raise PipelineStop(
            f"Time budget exhausted before stage '{stage_name}'. Progress saved to {paths.run_state_path}."
        )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    original_columns = list(normalized.columns)
    lower_map = {str(col).lower(): col for col in normalized.columns}
    rename_map: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        canonical_present = canonical in normalized.columns
        for alias in aliases:
            src = lower_map.get(alias.lower())
            if src is None:
                continue
            if src == canonical:
                canonical_present = True
                break
            if not canonical_present:
                rename_map[src] = canonical
                canonical_present = True
                break
    if rename_map:
        normalized = normalized.rename(columns=rename_map)

    if "pIC50" in normalized.columns:
        normalized["pIC50"] = pd.to_numeric(normalized["pIC50"], errors="coerce")
    if "ic50_nm_median" in normalized.columns:
        normalized["ic50_nm_median"] = pd.to_numeric(normalized["ic50_nm_median"], errors="coerce")
    for col in ["target", "uniprot_id", "pdb_id", "smiles", "inchikey"]:
        if col in normalized.columns:
            normalized[col] = normalized[col].astype(str).str.strip()
    log.debug("Column normalization: %s -> %s", original_columns, list(normalized.columns))
    return normalized


def validate_environment(debug_mode: bool = False) -> dict[str, Any]:
    report = {
        "python_version": sys.version,
        "platform": sys.platform,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device_default": DEVICE,
        "imports": {},
    }
    for package_name in ["rdkit", "pyarrow", "streamlit", "torch_geometric"]:
        try:
            __import__(package_name)
            report["imports"][package_name] = "ok"
        except Exception as exc:
            report["imports"][package_name] = f"missing: {exc}"
    if debug_mode:
        log.info("Environment report: %s", report)
    return report


def validate_required_files(paths: PipelinePaths, args: argparse.Namespace) -> dict[str, bool]:
    cacheable_targets = {
        "dataset": paths.dataset.exists(),
        "retained_targets": paths.retained_targets.exists(),
        "registry": paths.registry.exists(),
        "ligand_store": paths.ligand_store.exists(),
        "protein_store": paths.protein_store.exists(),
    }

    if not paths.excel.exists() and not cacheable_targets["dataset"]:
        raise PipelineStop(
            f"Input registry workbook not found at {paths.excel}. "
            "Provide --excel or ensure cached dataset artefacts already exist."
        )

    return cacheable_targets


def diagnose_pipeline(paths: PipelinePaths, config_ids: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "paths": {k: str(v) for k, v in asdict(paths).items()},
        "dataset_exists": paths.dataset.exists(),
        "retained_targets_exists": paths.retained_targets.exists(),
        "protein_store_exists": paths.protein_store.exists(),
        "ligand_store_exists": paths.ligand_store.exists(),
        "results_exists": paths.results_dir.exists(),
        "gpu_available": torch.cuda.is_available(),
        "requested_configs": config_ids,
    }

    if paths.dataset.exists():
        try:
            df = normalize_columns(pd.read_parquet(paths.dataset))
            report["dataset_rows"] = int(len(df))
            report["dataset_columns"] = list(df.columns)
            report["dataset_missing_required_columns"] = sorted(REQUIRED_DATASET_COLUMNS - set(df.columns))
            report["_dataset_frame_ready"] = df
        except Exception as exc:
            report["dataset_error"] = str(exc)

    if paths.ligand_store.exists():
        try:
            lig_store = load_ligand_feature_store(paths.ligand_store)
            report["ligand_feature_count"] = int(len(lig_store))
        except Exception as exc:
            report["ligand_store_error"] = str(exc)

    if paths.protein_store.exists():
        try:
            prot_store = load_protein_feature_store(paths.protein_store)
            report["protein_feature_count"] = int(len(prot_store))
            report["_protein_keys"] = list(prot_store.keys())
        except Exception as exc:
            report["protein_store_error"] = str(exc)

    checkpoint_status = {}
    for cid in config_ids:
        checkpoint_status[cid] = any(paths.checkpoint_dir.glob(f"{cid}_seed*.pt"))
    report["checkpoint_status"] = checkpoint_status
    if "_dataset_frame_ready" in report:
        df = report.pop("_dataset_frame_ready")
        ligand_keys = None
        protein_keys = None
        if paths.ligand_store.exists():
            try:
                lig_store = load_ligand_feature_store(paths.ligand_store)
                ligand_keys = set(lig_store.keys()) if hasattr(lig_store, "keys") else set(lig_store)
            except Exception:
                ligand_keys = None
        if "_protein_keys" in report:
            protein_keys = set(report.pop("_protein_keys"))
        if ligand_keys is not None and "inchikey" in df.columns:
            report["ligand_feature_coverage_fraction"] = round(float(df["inchikey"].isin(ligand_keys).mean()), 4)
        if protein_keys is not None and "uniprot_id" in df.columns:
            report["protein_feature_coverage_fraction"] = round(float(df["uniprot_id"].isin(protein_keys).mean()), 4)
    _write_json(paths.diagnostics_path, report)
    return report


def filter_valid_uniprot_targets(paths: PipelinePaths) -> pd.DataFrame:
    if paths.retained_targets.exists():
        return pd.read_csv(paths.retained_targets)
    if paths.registry.exists():
        registry_df = pd.read_parquet(paths.registry)
        registry_df = normalize_columns(registry_df)
        if "protein_status" in registry_df.columns:
            registry_df = registry_df[registry_df["protein_status"] == "retained"].copy()
        keep_cols = [c for c in ["target", "uniprot_id", "pdb_id", "protein_status", "feature_status"] if c in registry_df.columns]
        return registry_df[keep_cols].drop_duplicates(subset=["uniprot_id"])
    return pd.DataFrame(columns=["target", "uniprot_id", "pdb_id"])


def build_dataset_stage(paths: PipelinePaths, args: argparse.Namespace) -> pd.DataFrame:
    should_rebuild = (not args.use_cache) or (not paths.dataset.exists())
    if should_rebuild:
        if not paths.excel.exists():
            raise PipelineStop("Dataset build requested but the Excel registry file is missing.")
        log.info("Building dataset artefacts...")
        build_dataset(
            excel_path=str(paths.excel),
            out_dir=paths.work_dir,
            ratio_mode=args.ratio_mode,
            seed=args.seed,
            use_cache=args.use_cache,
        )
    if not paths.dataset.exists():
        raise PipelineStop(f"Dataset artefact missing after build at {paths.dataset}")
    df = pd.read_parquet(paths.dataset)
    df = normalize_columns(df)
    return df


def validate_dataset_schema(df: pd.DataFrame, paths: PipelinePaths) -> pd.DataFrame:
    df = normalize_columns(df)
    if df.empty:
        raise PipelineStop("Dataset is empty after load/build.")
    missing = REQUIRED_DATASET_COLUMNS - set(df.columns)
    if missing:
        raise PipelineStop(f"Dataset is missing required columns after normalization: {sorted(missing)}")
    if "pIC50_std" not in df.columns:
        df["pIC50_std"] = 0.5
    if "activity_label" not in df.columns and "pIC50" in df.columns:
        df["activity_label"] = np.where(df["pIC50"] >= 7.0, "active", "inactive")  # type: ignore[name-defined]
    df.to_parquet(paths.dataset, index=False)
    return df


def build_or_load_protein_features(
    paths: PipelinePaths,
    args: argparse.Namespace,
    retained_targets: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    if args.use_cache and paths.protein_store.exists():
        log.info("Loading cached protein feature store from %s", paths.protein_store)
        return load_protein_feature_store(paths.protein_store)

    if retained_targets is None or retained_targets.empty:
        log.warning("Protein feature build deferred: retained target list is not available yet.")
        return {}

    log.info("Building protein feature store...")
    return build_feature_store(
        retained_targets=retained_targets,
        store_path=paths.protein_store,
        use_cache=args.use_cache,
    )


def build_ligand_features(paths: PipelinePaths, args: argparse.Namespace) -> Any:
    if args.use_cache and paths.ligand_store.exists():
        log.info("Loading cached ligand feature store from %s", paths.ligand_store)
        return load_ligand_feature_store(paths.ligand_store)

    if not paths.dataset.exists():
        raise PipelineStop("Cannot build ligand features because dataset_clean.parquet is missing.")

    log.info("Building ligand feature store...")
    build_ligand_feature_store(
        dataset_path=str(paths.dataset),
        output_path=str(paths.ligand_store),
        compute_3d=args.compute_3d,
        conformer_seed=args.seed,
        use_cache=args.use_cache,
    )
    return load_ligand_feature_store(paths.ligand_store)


def validate_feature_alignment(
    df: pd.DataFrame,
    ligand_store: Any,
    protein_store: dict[str, Any],
    paths: PipelinePaths,
    min_rows: int = MIN_ALIGNED_ROWS,
) -> pd.DataFrame:
    if df.empty:
        raise PipelineStop("Cannot align features because dataset is empty.")

    ligand_keys = set(ligand_store.keys()) if hasattr(ligand_store, "keys") else set(ligand_store)
    protein_keys = set(protein_store.keys())
    before_rows = len(df)
    aligned = df[df["inchikey"].isin(ligand_keys)].copy()
    ligand_rows = len(aligned)
    aligned = aligned[aligned["uniprot_id"].isin(protein_keys)].copy()
    after_rows = len(aligned)

    report = {
        "rows_before_alignment": before_rows,
        "rows_after_ligand_alignment": ligand_rows,
        "rows_after_protein_alignment": after_rows,
        "ligand_coverage_fraction": round(ligand_rows / before_rows, 4) if before_rows else 0.0,
        "protein_coverage_fraction": round(after_rows / ligand_rows, 4) if ligand_rows else 0.0,
        "unique_ligands_after_alignment": int(aligned["inchikey"].nunique()) if not aligned.empty else 0,
        "unique_targets_after_alignment": int(aligned["uniprot_id"].nunique()) if not aligned.empty else 0,
    }
    _write_json(paths.alignment_report_path, report)
    if after_rows < min_rows:
        raise PipelineStop(
            f"Aligned dataset has only {after_rows} rows after intersecting ligand and protein stores. "
            f"Minimum required rows = {min_rows}. See {paths.alignment_report_path}."
        )
    aligned.to_parquet(paths.work_dir / "dataset_aligned.parquet", index=False)
    return aligned


def _checkpoints_exist(paths: PipelinePaths, config_ids: list[str], seeds: list[int]) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for cid in config_ids:
        files = []
        for seed in seeds:
            ckpt = paths.checkpoint_dir / f"{cid}_seed{seed}.pt"
            if ckpt.exists():
                files.append(str(ckpt))
        hits[cid] = files
    return hits


def train_models(
    aligned_df: pd.DataFrame,
    ligand_store: Any,
    protein_store: dict[str, Any],
    paths: PipelinePaths,
    args: argparse.Namespace,
    config_ids: list[str],
    start_time: float,
) -> dict[str, Any]:
    if args.skip_training:
        log.info("Skipping training because --skip_training was requested.")
        return {}

    checkpoint_hits = _checkpoints_exist(paths, config_ids, [args.seed])
    if args.use_cache and all(checkpoint_hits.get(cid) for cid in config_ids):
        log.info("All requested checkpoints already exist; skipping training because --use_cache is enabled.")
        return {}

    dataset = KinaseLigandDataset(aligned_df, ligand_store, protein_store)
    batch_size = args.batch_size or (64 if str(args.device).startswith("cuda") else 24)
    train_cfg = TrainConfig(
        seeds=[args.seed],
        epochs=args.epochs,
        batch_size=batch_size,
        patience=args.patience,
        num_workers=args.num_workers,
        use_tensorboard=False,
        checkpoint_dir=str(paths.checkpoint_dir),
        global_start_time=start_time,
        global_time_limit_hours=args.time_limit_hours,
    )
    log.info("Training configs: %s", config_ids)
    return train_all_configs(dataset=dataset, train_cfg=train_cfg, config_ids=config_ids, device=args.device)


def validate_checkpoints(paths: PipelinePaths, config_ids: list[str], seeds: list[int]) -> dict[str, list[str]]:
    hits = _checkpoints_exist(paths, config_ids, seeds)
    missing = [cid for cid, files in hits.items() if not files]
    if missing:
        raise PipelineStop(f"Checkpoint validation failed. Missing checkpoints for: {missing}")
    return hits


def run_evaluation(
    aligned_df: pd.DataFrame,
    ligand_store: Any,
    protein_store: dict[str, Any],
    paths: PipelinePaths,
    args: argparse.Namespace,
    config_ids: list[str],
) -> dict[str, Any]:
    all_eval: dict[str, Any] = {}
    for cid in config_ids:
        try:
            eval_result = run_full_evaluation(
                config_id=cid,
                seeds=[args.seed],
                dataset_df=aligned_df,
                ligand_store=ligand_store,
                protein_store=protein_store,
                checkpoint_dir=str(paths.checkpoint_dir),
                threshold=args.threshold,
                device=args.device,
            )
            all_eval[cid] = eval_result
        except Exception as exc:
            log.error("Evaluation failed for %s: %s", cid, exc, exc_info=True)
            all_eval[cid] = {"error": str(exc)}

    results_csv = paths.results_dir / "results.csv"
    per_kinase_csv = paths.results_dir / "per_kinase_results.csv"
    build_results_csv(all_eval, output_path=str(results_csv))
    save_per_kinase_results(all_eval, output_path=str(per_kinase_csv))
    return all_eval


def run_experiments(paths: PipelinePaths, args: argparse.Namespace, config_ids: list[str]) -> dict[str, Any]:
    exp_cfg = ExperimentConfig(
        config_ids=config_ids,
        seeds=[args.seed],
        checkpoint_dir=str(paths.checkpoint_dir),
        results_dir=str(paths.results_dir),
        threshold=args.threshold,
        device=args.device,
        skip_training=args.skip_training,
        fast_debug=args.debug_mode,
    )
    exp_cfg.train_cfg.checkpoint_dir = str(paths.checkpoint_dir)
    exp_cfg.train_cfg.seeds = [args.seed]
    exp_cfg.train_cfg.epochs = min(args.epochs, 4) if args.debug_mode else args.epochs
    exp_cfg.train_cfg.batch_size = args.batch_size or (64 if str(args.device).startswith("cuda") else 24)
    exp_cfg.train_cfg.patience = min(args.patience, 2) if args.debug_mode else args.patience
    exp_cfg.train_cfg.num_workers = args.num_workers

    return run_time_bounded_ablation(
        dataset_path=str(paths.dataset),
        lig_store_path=str(paths.ligand_store),
        prot_store_path=str(paths.protein_store),
        exp_cfg=exp_cfg,
        time_limit_hours=args.time_limit_hours,
    )


def launch_streamlit(paths: PipelinePaths) -> None:
    module_path = Path("module10_streamlit.py")
    if not module_path.exists():
        raise PipelineStop("Cannot launch Streamlit because module10_streamlit.py is missing.")
    log.info("Launching Streamlit app...")
    subprocess.Popen(["streamlit", "run", str(module_path)], cwd=str(paths.work_dir.parent))


def _default_config_ids(config_arg: str) -> list[str]:
    if config_arg == "all":
        return list(PRIORITY_CONFIGS)
    parsed = [c.strip() for c in config_arg.split(",") if c.strip()]
    invalid = [c for c in parsed if c not in ALL_CONFIG_IDS]
    if invalid:
        raise PipelineStop(f"Unknown config id(s): {invalid}. Valid configs: {sorted(ALL_CONFIG_IDS)}")
    return parsed


def _should_stop_after(stage: str, only_stage: Optional[str]) -> bool:
    return only_stage == stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crash-resistant master entrypoint for the kinase-ligand pipeline.")
    parser.add_argument("--excel", default="ML.xlsx", help="Input registry workbook")
    parser.add_argument("--work-dir", default="./pipeline_outputs", help="Shared output root")
    parser.add_argument("--config", default="all", help="Comma-separated config IDs or 'all'")
    parser.add_argument("--seed", type=int, default=42, help="Primary random seed")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--time_limit_hours", type=float, default=24.0, help="Wall-clock budget in hours")
    parser.add_argument("--ratio_mode", default="1:1", choices=["1:1", "1:2", "1:3"], help="Active:inactive ratio")
    parser.add_argument("--device", default=DEVICE, help="Training/inference device")
    parser.add_argument("--threshold", type=float, default=7.0, help="pIC50 activity threshold")
    parser.add_argument("--compute_3d", action="store_true", help="Enable 3D ligand conformers")
    parser.add_argument("--use_cache", action="store_true", help="Reuse existing artefacts where valid")
    parser.add_argument("--skip_training", action="store_true", help="Skip model training")
    parser.add_argument("--only_stage", choices=["dataset", "features", "train", "eval", "app"], default=None, help="Run only through a specific stage")
    parser.add_argument("--debug_mode", action="store_true", help="Enable extra logging and shorter experiment settings")
    parser.add_argument("--launch_streamlit", action="store_true", help="Launch the Streamlit app after a successful run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_ids = _default_config_ids(args.config)
    paths = build_paths(args.work_dir, args.excel)
    start_time = time.time()

    try:
        env_report = validate_environment(debug_mode=args.debug_mode)
        validate_required_files(paths, args)
        diag_report = diagnose_pipeline(paths, config_ids)
        _write_json(
            paths.run_state_path,
            {
                "status": "started",
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
                "environment": env_report,
                "diagnostics": diag_report,
            },
        )

        _assert_time_budget("build_or_load_protein_features", start_time, args.time_limit_hours, paths)
        protein_store = build_or_load_protein_features(paths, args, retained_targets=None)

        _assert_time_budget("filter_valid_uniprot_targets", start_time, args.time_limit_hours, paths)
        retained_targets = filter_valid_uniprot_targets(paths)

        _assert_time_budget("build_dataset", start_time, args.time_limit_hours, paths)
        dataset_df = build_dataset_stage(paths, args)

        _assert_time_budget("validate_dataset_schema", start_time, args.time_limit_hours, paths)
        dataset_df = validate_dataset_schema(dataset_df, paths)
        retained_targets = filter_valid_uniprot_targets(paths)
        if _should_stop_after("dataset", args.only_stage):
            log.info("Stopping after dataset stage by request.")
            return

        _assert_time_budget("build_ligand_features", start_time, args.time_limit_hours, paths)
        ligand_store = build_ligand_features(paths, args)

        _assert_time_budget("finalize_protein_features", start_time, args.time_limit_hours, paths)
        if not protein_store:
            protein_store = build_or_load_protein_features(paths, args, retained_targets=retained_targets)
        if not protein_store:
            raise PipelineStop("Protein feature store is unavailable after attempted build/load.")

        _assert_time_budget("validate_feature_alignment", start_time, args.time_limit_hours, paths)
        aligned_df = validate_feature_alignment(dataset_df, ligand_store, protein_store, paths)
        if _should_stop_after("features", args.only_stage):
            log.info("Stopping after feature stages by request.")
            return

        _assert_time_budget("train_models", start_time, args.time_limit_hours, paths)
        train_models(aligned_df, ligand_store, protein_store, paths, args, config_ids, start_time)
        if _should_stop_after("train", args.only_stage):
            log.info("Stopping after training stage by request.")
            return

        _assert_time_budget("validate_checkpoints", start_time, args.time_limit_hours, paths)
        validate_checkpoints(paths, config_ids, [args.seed])

        _assert_time_budget("run_evaluation", start_time, args.time_limit_hours, paths)
        run_evaluation(aligned_df, ligand_store, protein_store, paths, args, config_ids)
        if _should_stop_after("eval", args.only_stage):
            log.info("Stopping after evaluation stage by request.")
            return

        _assert_time_budget("run_experiments", start_time, args.time_limit_hours, paths)
        run_experiments(paths, args, config_ids)

        if args.launch_streamlit or args.only_stage == "app":
            launch_streamlit(paths)

        _write_json(
            paths.run_state_path,
            {
                "status": "completed",
                "elapsed_seconds": round(time.time() - start_time, 2),
                "work_dir": str(paths.work_dir),
                "results_dir": str(paths.results_dir),
            },
        )
        log.info("Pipeline completed successfully. Outputs are in %s", paths.work_dir.resolve())

    except PipelineStop as exc:
        log.warning(str(exc))
        _write_json(
            paths.run_state_path,
            {
                "status": "stopped",
                "reason": str(exc),
                "elapsed_seconds": round(time.time() - start_time, 2),
            },
        )
        raise SystemExit(1) from exc
    except Exception as exc:
        log.error("Unhandled pipeline failure: %s", exc, exc_info=True)
        _write_json(
            paths.run_state_path,
            {
                "status": "failed",
                "reason": str(exc),
                "elapsed_seconds": round(time.time() - start_time, 2),
            },
        )
        raise


if __name__ == "__main__":
    main()
