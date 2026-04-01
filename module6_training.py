"""
module6_training.py
===================
TITLE
Model training engine for the kinase-ligand pipeline.

PURPOSE
This module trains the configured neural models against the aligned kinase-
ligand dataset while guarding against non-finite losses, bad batches, and
checkpoint gaps.

WHAT IT DOES
- Builds datasets and splits for training.
- Trains regression and classification model variants.
- Saves reproducible checkpoints and split indices.
- Runs guarded prediction helpers reused by later modules.

HOW IT WORKS
1. Load normalized dataset rows plus ligand/protein feature stores.
2. Build train/validation splits.
3. Train one or more configs across requested seeds.
4. Skip unsafe batches and keep detailed logs.
5. Save best and last checkpoints.

INPUT CONTRACT
- Dataset rows aligned to ligand and protein feature stores.
- Valid model config IDs and training hyperparameters.

OUTPUT CONTRACT
- Checkpoints, split-index JSON files, and prediction utilities.

DEPENDENCIES
- torch, torch_geometric, pandas, numpy
- module5_models.py

CRITICAL ASSUMPTIONS
- Input feature stores are already aligned to the dataset.
- Config IDs are valid and compatible with the feature schema.

FAILURE MODES
- Shape mismatches during forward passes
- NaN or Inf losses
- Empty trainable subset for a requested config
- Missing checkpoints after training

SAFETY CHECKS IMPLEMENTED
- Batch sanitization
- Safe forward wrappers with skip-and-log behaviour
- Non-finite loss and gradient guards
- Best and last checkpoint guarantees

HOW TO RUN
- `python module6_training.py --dataset ./pipeline_outputs/dataset_clean.parquet --config full_model --seeds 42`

HOW IT CONNECTS TO PIPELINE
It is the main training backend used directly by `run_pipeline.py` and indirectly
by experiments, evaluation, and uncertainty modules.
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
import json
import platform
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch_geometric.data import Batch as PyGBatch, Data
from torch.utils.data import DataLoader, Dataset, Subset
from progress_utils import progress_iter

from module5_models import (
    ALL_CONFIG_IDS,
    BaseModel,
    ModelConfig,
    build_model,
    get_model_config,
    PHYSCHEM_DIM,
    ATOM_FEAT_DIM,
    BOND_FEAT_DIM,
)

log = logging.getLogger("module6")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Training hyper-parameters (overridable via TrainConfig)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SEEDS: list[int] = [42]
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

SMILES_CHARSET = list("CNOPSFIBrclhn1234567890()=#+-[]\\/ ")
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
char_to_int = {"<PAD>": PAD_IDX, "<SOS>": SOS_IDX, "<EOS>": EOS_IDX, "<UNK>": UNK_IDX}
char_to_int.update({ch: i + 4 for i, ch in enumerate(SMILES_CHARSET)})
MAX_LEN = 120


@dataclass
class TrainConfig:
    """
    Top-level training configuration.

    Attributes
    ----------
    seeds            : list of random seeds for multi-seed training
    epochs           : number of training epochs
    batch_size       : molecules per batch
    lr               : initial learning rate
    weight_decay     : AdamW weight decay
    grad_clip        : max gradient norm (1.0 recommended)
    use_sample_weight: weight each sample by 1/(pIC50_std + eps)
    weight_eps       : epsilon added to pIC50_std before inversion
    val_fraction     : fraction of data held out for validation
    test_fraction    : fraction of data held out for test
    split_strategy   : 'scaffold' | 'random'
    checkpoint_dir   : directory to save model checkpoints
    results_path     : path to accumulate results CSV
    use_tensorboard  : log to TensorBoard if available
    patience         : early-stopping patience (epochs without val improvement)
    min_delta        : minimum improvement to reset patience counter
    scheduler        : 'cosine' | 'plateau' | 'none'
    warmup_epochs    : linear LR warm-up epochs
    cvae_beta_warmup : number of epochs to linearly ramp CVAE KL weight to 1
    cvae_recon_weight: weight on reconstruction loss relative to KL
    """
    seeds:              list[int] = field(default_factory=lambda: [42])
    epochs:             int   = 30
    batch_size:         int   = 64 if torch.cuda.is_available() else 24
    lr:                 float = 3e-4
    weight_decay:       float = 1e-4
    grad_clip:          float = 1.0
    use_sample_weight:  bool  = True
    weight_eps:         float = 1e-3
    val_fraction:       float = 0.1
    test_fraction:      float = 0.1
    split_strategy:     str   = "scaffold"
    checkpoint_dir:     str   = "./checkpoints"
    results_path:       str   = "results.csv"
    use_tensorboard:    bool  = False
    patience:           int   = 8
    min_delta:          float = 1e-4
    scheduler:          str   = "cosine"
    warmup_epochs:      int   = 3
    cvae_beta_warmup:   int   = 30
    cvae_recon_weight:  float = 1.0
    family:             str   = "regression"
    label_scheme:       str   = "pIC50_continuous"
    task_type_override: Optional[str] = None
    active_cutoff_nm:   float = 500.0
    inactive_cutoff_nm: float = 5000.0
    binary_threshold_pic50: float = 7.0
    use_class_balance:  bool  = True
    min_weight_std:     float = 0.05
    max_batch_weight:   float = 5.0
    num_workers:        Optional[int] = None
    prefetch_factor:    int   = 2
    persistent_workers: bool  = True
    pin_memory:         Optional[bool] = None
    use_amp:            Optional[bool] = None
    global_start_time:  Optional[float] = None
    global_time_limit_hours: float = 24.0
    min_protein_pocket_confidence: float = 0.5


def _recommended_num_workers(train_cfg: TrainConfig) -> int:
    if train_cfg.num_workers is not None:
        return max(0, int(train_cfg.num_workers))

    cpu_count = os.cpu_count() or 4
    if platform.system() == "Windows":
        return min(4, max(1, cpu_count // 4))
    return min(8, max(2, cpu_count // 2))


def _build_loader_kwargs(train_cfg: TrainConfig, is_train: bool) -> dict:
    num_workers = _recommended_num_workers(train_cfg)
    pin_memory = torch.cuda.is_available() if train_cfg.pin_memory is None else bool(train_cfg.pin_memory)

    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(train_cfg.persistent_workers)
        kwargs["prefetch_factor"] = max(1, int(train_cfg.prefetch_factor))
    if is_train:
        kwargs["drop_last"] = True
    return kwargs


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Dataset
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class KinaseLigandDataset(Dataset):
    """
    PyTorch Dataset wrapping the clean dataset parquet + feature stores.

    Each item returns a dict with:
        graph_data   : torch_geometric.data.Data object (ligand graph)
        physchem     : (PHYSCHEM_DIM,)   float tensor
        esm_pocket   : (85, 1280)        float tensor
        confidence   : (85,)             float tensor
        pIC50        : scalar float
        pIC50_std    : scalar float (always finite â€” guaranteed by module1)
        weight       : scalar float (inverse uncertainty weight)
        inchikey     : str
        uniprot_id   : str
        smiles       : str
    """

    def __init__(
        self,
        dataset_df:       pd.DataFrame,
        ligand_store:     dict,           # {inchikey: torch_geometric.data.Data}
        protein_store:    dict,           # {uniprot_id: ProteinFeatures}
        weight_eps:       float = 1e-3,
        min_weight_std:   float = 0.05,
        use_sample_weight: bool = True,
        min_protein_pocket_confidence: float = 0.5,
    ) -> None:
        super().__init__()
        self.weight_eps        = weight_eps
        self.min_weight_std    = min_weight_std
        self.use_sample_weight = use_sample_weight
        self.min_protein_pocket_confidence = float(min_protein_pocket_confidence)

        # Assign stores first â€” they are referenced immediately below
        self.ligand_store  = ligand_store
        self.protein_store = protein_store

        # Filter to rows where both ligand and protein features exist
        ligand_store_keys  = set(ligand_store.keys())
        protein_store_keys = set(protein_store.keys())
        availability_mask = (
            dataset_df["inchikey"].isin(ligand_store_keys) &
            dataset_df["uniprot_id"].isin(protein_store_keys)
        )
        pocket_conf_series = dataset_df["uniprot_id"].map(
            lambda uid: float(getattr(protein_store.get(uid), "pocket_confidence", 0.0))
        )
        confidence_mask = pocket_conf_series >= self.min_protein_pocket_confidence
        mask = availability_mask & confidence_mask
        n_dropped = (~mask).sum()
        if n_dropped > 0:
            log.warning(
                "KinaseLigandDataset: dropped %d rows because ligand/protein features "
                "were missing or pocket confidence was below %.2f.",
                n_dropped,
                self.min_protein_pocket_confidence,
            )
        self.df = dataset_df[mask].reset_index(drop=True)

        log.info(
            "Dataset: %d valid samples (%d unique ligands, %d unique targets) | weight_std_floor=%.4f",
            len(self.df),
            self.df["inchikey"].nunique(),
            self.df["uniprot_id"].nunique(),
            self.min_weight_std,
        )

        # Pre-validate pIC50 and pIC50_std are NaN-free
        assert self.df["pIC50"].isna().sum() == 0,     "NaN in pIC50 â€” abort"
        assert self.df["pIC50_std"].isna().sum() == 0, "NaN in pIC50_std â€” abort"

        # Compute global weight mean for normalization
        if self.use_sample_weight:
            effective_std = self.df["pIC50_std"].clip(lower=self.min_weight_std)
            weights = 1.0 / (effective_std + self.weight_eps)
            self.global_weight_mean = weights.mean()
        else:
            self.global_weight_mean = 1.0

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        lig_data = self.ligand_store[row["inchikey"]]
        prot     = self.protein_store[row["uniprot_id"]]

        pic50     = float(row["pIC50"])
        pic50_std = float(row["pIC50_std"])

        # Sample weight: semi-empirical measurement frequency scaling
        if self.use_sample_weight:
            n_meas = float(row.get("n_measurements", 1.0))
            effective_std = max(pic50_std, self.min_weight_std)
            weight = math.sqrt(n_meas) / (effective_std + self.weight_eps)
        else:
            weight = 1.0

        smi = row["smiles"]
        pocket_confidence = float(getattr(prot, "pocket_confidence", 0.0))
        weight *= pocket_confidence

        tokens = [SOS_IDX] + \
                 [char_to_int.get(c, UNK_IDX) for c in smi][:118] + \
                 [EOS_IDX]

        seq = torch.full((120,), PAD_IDX, dtype=torch.long)
        seq[:len(tokens)] = torch.tensor(tokens, dtype=torch.long)

        return {
            "graph_data":  lig_data,
            "physchem":    lig_data.physchem,                    # (PHYSCHEM_DIM,)
            "morgan_fp":   getattr(lig_data, "morgan_fp",
                                   torch.zeros(1024)),           # (1024,)
            "esm_pocket":  prot.esm_pocket,                      # (85, 1280)
            "confidence":  prot.confidence,                      # (85,)
            "pocket_confidence": torch.tensor(pocket_confidence, dtype=torch.float),
            "pIC50":       torch.tensor(pic50,     dtype=torch.float),
            "pIC50_std":   torch.tensor(pic50_std, dtype=torch.float),
            "ic50_nm":     torch.tensor(float(row.get("ic50_nm_median", 10 ** (9 - pic50))), dtype=torch.float),
            "label":       torch.tensor(
                               float(row["classification_label"])
                               if "classification_label" in row and pd.notna(row["classification_label"])
                               else float("nan"),
                               dtype=torch.float,
                           ),
            "weight":      torch.tensor(weight,    dtype=torch.float),
            "inchikey":    row["inchikey"],
            "uniprot_id":  row["uniprot_id"],
            "smiles":      row["smiles"],
            "target_seq":  seq,
        }

    def get_smiles_list(self) -> list[str]:
        return self.df["smiles"].tolist()

    def get_uniprot_ids(self) -> list[str]:
        return self.df["uniprot_id"].tolist()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Collate function: PyG graph batching + dense tensor stacking
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def collate_fn(items: list[dict]) -> dict:
    """
    Custom collate that handles heterogeneous batch items:
    - PyG Data objects are batched via PyGBatch.from_data_list.
    - Dense tensors are stacked normally.
    - Scalar tensors are stacked to (B,).
    """
    graph_data_list = [
        Data(
            x=item["graph_data"].x,
            edge_index=item["graph_data"].edge_index,
            edge_attr=item["graph_data"].edge_attr,
        )
        for item in items
    ]
    pyg_batch = PyGBatch.from_data_list(graph_data_list)

    batch = {
        # Graph (PyG batch)
        "x":          pyg_batch.x,
        "edge_index": pyg_batch.edge_index,
        "edge_attr":  pyg_batch.edge_attr,
        "batch":      pyg_batch.batch,

        # Dense features
        "physchem":   torch.stack([i["physchem"]   for i in items]),
        "morgan_fp":  torch.stack([i["morgan_fp"]  for i in items]),
        "esm_pocket": torch.stack([i["esm_pocket"] for i in items]),
        "confidence": torch.stack([i["confidence"] for i in items]),
        "pocket_confidence": torch.stack([i["pocket_confidence"] for i in items]),

        # Targets
        "pIC50":      torch.stack([i["pIC50"]     for i in items]),
        "pIC50_std":  torch.stack([i["pIC50_std"] for i in items]),
        "ic50_nm":    torch.stack([i["ic50_nm"]   for i in items]),
        "label":      torch.stack([i["label"]     for i in items]),
        "weight":     torch.stack([i["weight"]    for i in items]),

        # CVAE teacher-forcing sequence (always included; all-PAD for non-CVAE)
        "target_seq": torch.stack([i["target_seq"] for i in items]),

        # Protein padding mask derived from confidence (0 = padded position)
        "protein_mask": (torch.stack([i["confidence"] for i in items]) == 0.0),

        # Meta
        "inchikey":   [i["inchikey"]   for i in items],
        "uniprot_id": [i["uniprot_id"] for i in items],
        "smiles":     [i["smiles"]     for i in items],
    }
    return batch


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Murcko scaffold split
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def murcko_scaffold_split(
    smiles_list: list[str],
    val_frac:    float = 0.1,
    test_frac:   float = 0.1,
    seed:        int   = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split dataset indices using Bemis-Murcko scaffold grouping.

    Scaffolds are sorted by frequency (most common first); scaffolds are
    assigned greedily to train/val/test to respect the target fractions
    while ensuring no scaffold appears in more than one split.

    Returns
    -------
    (train_indices, val_indices, test_indices)
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffold_to_indices: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            ) if mol else ""
        except Exception:
            scaffold = ""
        if not scaffold:
            scaffold = smi   # fallback unique grouping
        scaffold_to_indices.setdefault(scaffold, []).append(i)

    # Sort scaffolds: rarest first (put them in train) â€” prevents
    # test contamination from popular scaffolds
    rng = random.Random(seed)
    scaffold_groups = list(scaffold_to_indices.values())
    rng.shuffle(scaffold_groups)
    scaffold_groups.sort(key=len)   # rarest scaffolds first

    n_total = len(smiles_list)
    n_val   = int(math.ceil(n_total * val_frac))
    n_test  = int(math.ceil(n_total * test_frac))

    train_idx, val_idx, test_idx = [], [], []
    for group in scaffold_groups:
        if len(test_idx) < n_test:
            test_idx.extend(group)
        elif len(val_idx) < n_val:
            val_idx.extend(group)
        else:
            train_idx.extend(group)

    log.info(
        "Scaffold split: train=%d  val=%d  test=%d  (total=%d)",
        len(train_idx), len(val_idx), len(test_idx), n_total,
    )
    return train_idx, val_idx, test_idx


def random_split(
    n: int,
    val_frac:  float = 0.1,
    test_frac: float = 0.1,
    seed:      int   = 42,
) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n).tolist()
    n_val  = int(math.ceil(n * val_frac))
    n_test = int(math.ceil(n * test_frac))
    return indices[n_val + n_test:], indices[:n_val], indices[n_val: n_val + n_test]


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Loss functions
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_GNLL = nn.GaussianNLLLoss(full=True, eps=1e-6, reduction="none")
_BCE = nn.BCEWithLogitsLoss(reduction="none")


def regression_loss(
    mu:      torch.Tensor,    # (B, 1)
    log_var: torch.Tensor,    # (B, 1)
    target:  torch.Tensor,    # (B,)
    weight:  Optional[torch.Tensor] = None,  # (B,)
    max_weight: Optional[float] = None,
) -> torch.Tensor:
    """
    Numerically stable heteroscedastic regression loss.

    Uses torch.nn.GaussianNLLLoss with var = exp(log_var).
    The GaussianNLLLoss internally clamps var >= eps=1e-6.

    Optional per-sample weighting by inverse measurement uncertainty.

    Returns scalar loss.
    """
    mu_sq     = mu.squeeze(-1)           # (B,)
    var       = torch.exp(log_var.squeeze(-1))  # (B,)  always positive

    # Validate â€” fail loudly rather than silently propagate NaN
    if torch.isnan(mu_sq).any() or torch.isinf(mu_sq).any():
        raise RuntimeError("NaN/Inf detected in model output mu â€” check model.")
    if torch.isnan(var).any() or torch.isinf(var).any():
        raise RuntimeError("NaN/Inf detected in predicted variance â€” check log_var clamping.")
    if torch.isnan(target).any():
        raise RuntimeError("NaN detected in regression targets â€” check dataset.")

    per_sample_loss = _GNLL(mu_sq, target, var)   # (B,)

    if weight is not None:
        # Normalise weights to mean=1 within this batch for scale-invariance
        w = weight / (weight.mean() + 1e-8)
        if max_weight is not None:
            w = w.clamp(max=max_weight)
        per_sample_loss = per_sample_loss * w

    return per_sample_loss.mean()


def cvae_elbo_loss(
    logits:    torch.Tensor,    # (B, T, VOCAB_SIZE)
    target_seq: torch.Tensor,   # (B, T)
    mu_z:      torch.Tensor,    # (B, d_z)
    log_var_z: torch.Tensor,    # (B, d_z)
    beta:      float = 1.0,
    recon_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CVAE Evidence Lower BOund loss.

    L_ELBO = L_recon + Î² Â· KL

    L_recon = CrossEntropy(logits, target_seq)  [teacher-forcing]
    KL      = -0.5 * sum(1 + log_var_z - mu_zÂ² - exp(log_var_z))

    Parameters
    ----------
    logits      : (B, T, VOCAB_SIZE)
    target_seq  : (B, T)  token indices (EOS-terminated, PAD-padded)
    mu_z        : (B, d_z)
    log_var_z   : (B, d_z)
    beta        : KL weight (0 â†’ pure reconstruction; ramp up during training)
    recon_weight: relative weight on reconstruction term

    Returns
    -------
    (total_loss, recon_loss, kl_loss) â€” all scalar tensors
    """
    target_out = target_seq[:, 1:]

    B, T_out, V = logits.shape
    # Reconstruction: flatten over batch and time
    recon = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX,
        reduction="mean"
    )(
        logits.reshape(B * T_out, V),
        target_out.reshape(B * T_out),
    )

    # KL divergence: standard ELBO scaling
    kl = -0.5 * torch.mean(
        torch.sum(1.0 + log_var_z - mu_z.pow(2) - log_var_z.exp(), dim=1)
    )

    total = recon_weight * recon + beta * kl
    return total, recon, kl


def classification_loss(
    logits: torch.Tensor,
    label: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logits = logits.squeeze(-1)
    label = label.float()

    if torch.isnan(logits).any() or torch.isinf(logits).any():
        raise RuntimeError("NaN/Inf detected in classification logits â€” check model.")
    if torch.isnan(label).any():
        raise RuntimeError("NaN detected in classification labels â€” check dataset.")

    if pos_weight is not None:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        return loss_fn(logits, label).mean()
    return _BCE(logits, label).mean()


def _classification_mode_for_config(model_cfg: ModelConfig) -> Optional[str]:
    if getattr(model_cfg, "label_scheme", "") == "IC50_500_5000_hard":
        return "hard_threshold"
    if getattr(model_cfg, "label_scheme", "") == "pIC50_7_binary":
        return "binary_threshold"
    if model_cfg.config_id == "classification_hard_threshold":
        return "hard_threshold"
    if model_cfg.task_type == "classification":
        return "binary_threshold"
    return None


def _effective_model_config(config_id: str, train_cfg: TrainConfig) -> ModelConfig:
    model_cfg = copy.deepcopy(get_model_config(config_id))
    if train_cfg.task_type_override is not None:
        model_cfg.task_type = train_cfg.task_type_override
    model_cfg.family = train_cfg.family
    model_cfg.label_scheme = train_cfg.label_scheme
    if model_cfg.task_type == "classification":
        model_cfg.use_uncertainty = False
        model_cfg.use_generative = False
    return model_cfg


def _eligible_indices_for_config(
    dataset: "KinaseLigandDataset",
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> list[int]:
    mode = _classification_mode_for_config(model_cfg)
    if mode == "hard_threshold":
        mask = (
            (dataset.df["ic50_nm_median"] < train_cfg.active_cutoff_nm) |
            (dataset.df["ic50_nm_median"] > train_cfg.inactive_cutoff_nm)
        )
        return dataset.df.index[mask].tolist()
    return dataset.df.index.tolist()


def _labels_from_batch(
    batch: dict,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> torch.Tensor:
    mode = _classification_mode_for_config(model_cfg)
    if mode == "hard_threshold":
        return (batch["ic50_nm"] < train_cfg.active_cutoff_nm).float()
    return (batch["pIC50"] >= train_cfg.binary_threshold_pic50).float()


def _compute_pos_weight(
    dataset: "KinaseLigandDataset",
    indices: list[int],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: str,
) -> Optional[torch.Tensor]:
    if model_cfg.task_type != "classification" or not train_cfg.use_class_balance:
        return None

    subset = dataset.df.iloc[indices]
    if _classification_mode_for_config(model_cfg) == "hard_threshold":
        labels = (subset["ic50_nm_median"] < train_cfg.active_cutoff_nm).astype(float)
    else:
        labels = (subset["pIC50"] >= train_cfg.binary_threshold_pic50).astype(float)

    n_pos = float(labels.sum())
    n_neg = float(len(labels) - n_pos)
    if n_pos <= 0 or n_neg <= 0:
        return None
    return torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Learning rate scheduling
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_scheduler(
    optimiser: torch.optim.Optimizer,
    cfg:       TrainConfig,
    n_train_steps: int,
) -> Optional[object]:
    """Build an LR scheduler based on TrainConfig.scheduler."""
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            T_max=cfg.epochs - cfg.warmup_epochs,
            eta_min=cfg.lr * 0.01,
        )
    elif cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=10, min_lr=cfg.lr * 0.01
        )
    elif cfg.scheduler == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: '{cfg.scheduler}'")


def warmup_lr(
    optimiser: torch.optim.Optimizer,
    epoch:     int,
    base_lr:   float,
    warmup_epochs: int,
) -> None:
    """Linear LR warm-up."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / max(warmup_epochs, 1)
        for pg in optimiser.param_groups:
            pg["lr"] = lr


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Seeding utility
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def set_seed(seed: int) -> None:
    """Fully deterministic seed (CPU + CUDA + Python + NumPy)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Single training run (one seed, one config)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def train_one_seed(
    model:         BaseModel,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    train_cfg:     TrainConfig,
    seed:          int,
    config_id:     str,
    pos_weight:    Optional[torch.Tensor] = None,
    device:        str = DEVICE,
) -> dict:
    """
    Train a single model for all epochs and return per-epoch metrics.

    Returns
    -------
    dict with keys:
        train_loss_history : list[float]
        val_loss_history   : list[float]
        best_val_loss      : float
        best_epoch         : int
        checkpoint_path    : str
    """
    set_seed(seed)
    model = model.to(device)

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = build_scheduler(optimiser, train_cfg, len(train_loader))
    use_amp = torch.cuda.is_available() and str(device).startswith("cuda") if train_cfg.use_amp is None else bool(train_cfg.use_amp)
    scaler = GradScaler("cuda", enabled=use_amp)

    is_cvae     = getattr(model.cfg, "use_generative", False)
    is_classification = model.cfg.task_type == "classification"
    ckpt_dir    = Path(train_cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path   = ckpt_dir / f"{config_id}_seed{seed}.pt"

    # TensorBoard (optional)
    writer = None
    if train_cfg.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(
                log_dir=str(ckpt_dir / "tensorboard" / f"{config_id}_seed{seed}")
            )
        except ImportError:
            log.info("TensorBoard not available â€” skipping.")

    best_val_loss     = float("inf")
    best_epoch        = 0
    patience_counter  = 0
    train_loss_hist: list[float] = []
    val_loss_hist:   list[float] = []
    skipped_batches_total = 0

    for epoch in progress_iter(
        range(train_cfg.epochs),
        total=train_cfg.epochs,
        desc=f"{config_id} seed {seed} epochs",
        leave=True,
    ):
        t0 = time.time()

        # â”€â”€ LR warm-up â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        warmup_lr(optimiser, epoch, train_cfg.lr, train_cfg.warmup_epochs)

        # â”€â”€ CVAE beta schedule â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if is_cvae:
            beta = min(1.0, epoch / max(train_cfg.cvae_beta_warmup, 1))
        else:
            beta = 0.0

        # â”€â”€ Training epoch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        model.train()
        train_losses: list[float] = []
        skipped_batches_epoch = 0

        for batch in progress_iter(
            train_loader,
            total=len(train_loader),
            desc=f"{config_id} seed {seed} train",
        ):
            batch = _sanitize_batch(_to_device(batch, device))

            optimiser.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_amp):
                if is_cvae:
                    outputs = _safe_forward(model, batch, return_cvae=True, context=f"train:{config_id}:seed{seed}:epoch{epoch+1}")
                    if outputs is None:
                        skipped_batches_epoch += 1
                        skipped_batches_total += 1
                        continue
                    mu, log_var, cvae_out = outputs
                    log_var = torch.clamp(log_var, min=-10.0, max=10.0)
                    reg_loss = regression_loss(
                        mu, log_var, batch["pIC50"],
                        weight=batch["weight"] if train_cfg.use_sample_weight else None,
                        max_weight=train_cfg.max_batch_weight,
                    )
                    if "target_seq" in batch and batch["target_seq"] is not None:
                        input_seq = batch["target_seq"][:, :-1]
                        target_out = batch["target_seq"][:, 1:]
                        elbo, recon, kl = cvae_elbo_loss(
                            cvae_out["logits"],
                            target_out,
                            cvae_out["mu_z"],
                            cvae_out["log_var_z"],
                            beta=beta,
                            recon_weight=train_cfg.cvae_recon_weight,
                        )
                        loss = reg_loss + elbo
                    else:
                        loss = reg_loss
                else:
                    if is_classification:
                        logits = _safe_forward(model, batch, context=f"train:{config_id}:seed{seed}:epoch{epoch+1}")
                        if logits is None:
                            skipped_batches_epoch += 1
                            skipped_batches_total += 1
                            continue
                        labels = _labels_from_batch(batch, model.cfg, train_cfg)
                        loss = classification_loss(
                            logits,
                            labels,
                            pos_weight=pos_weight,
                        )
                    else:
                        outputs = _safe_forward(model, batch, context=f"train:{config_id}:seed{seed}:epoch{epoch+1}")
                        if outputs is None:
                            skipped_batches_epoch += 1
                            skipped_batches_total += 1
                            continue
                        mu, log_var = outputs
                        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
                        loss = regression_loss(
                            mu, log_var, batch["pIC50"],
                            weight=batch["weight"] if train_cfg.use_sample_weight else None,
                            max_weight=train_cfg.max_batch_weight,
                        )

            # NaN guard on loss â€” never silently propagate
            if torch.isnan(loss) or torch.isinf(loss):
                skipped_batches_epoch += 1
                skipped_batches_total += 1
                optimiser.zero_grad(set_to_none=True)
                continue

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_cfg.grad_clip,
                    error_if_nonfinite=False,
                )
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    skipped_batches_epoch += 1
                    skipped_batches_total += 1
                    optimiser.zero_grad(set_to_none=True)
                    scaler.update()
                    continue

                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_cfg.grad_clip,
                    error_if_nonfinite=False,
                )
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    skipped_batches_epoch += 1
                    skipped_batches_total += 1
                    optimiser.zero_grad(set_to_none=True)
                    continue
                optimiser.step()

            train_losses.append(loss.item())

        if not train_losses:
            raise RuntimeError(
                f"[{config_id} seed={seed} epoch={epoch}] all training batches were skipped due to non-finite values."
            )
        train_loss = float(np.mean(train_losses))
        train_loss_hist.append(train_loss)

        # â”€â”€ Validation epoch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        val_metrics = _evaluate_loss(model, val_loader, train_cfg, device, beta, pos_weight=pos_weight)
        val_loss = val_metrics["total_loss"]
        val_loss_hist.append(val_loss)

        # â”€â”€ LR scheduler step â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if scheduler is not None and epoch >= train_cfg.warmup_epochs:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # â”€â”€ Early stopping & checkpoint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        monitor_loss = val_metrics["primary_loss"]
        if monitor_loss < best_val_loss - train_cfg.min_delta:
            best_val_loss    = monitor_loss
            best_epoch       = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch":        epoch,
                    "model_state":  model.state_dict(),
                    "optim_state":  optimiser.state_dict(),
                    "val_loss":     val_loss,
                    "config_id":    config_id,
                    "seed":         seed,
                    "model_config": model.cfg,
                },
                ckpt_path,
            )
        else:
            patience_counter += 1

        # â”€â”€ Logging â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        elapsed = time.time() - t0
        lr_now  = optimiser.param_groups[0]["lr"]
        total_elapsed = (
            time.time() - train_cfg.global_start_time
            if train_cfg.global_start_time is not None else elapsed
        )
        budget_left = max(
            0.0,
            train_cfg.global_time_limit_hours * 3600 - total_elapsed,
        ) if train_cfg.global_start_time is not None else float("nan")
        log.info(
            "[%s|seed=%d] Epoch %3d/%d | "
            "train=%.4f  val=%.4f  best=%.4f (ep%d)  "
            "lr=%.2e  patience=%d/%d  skipped=%d  "
            "run_elapsed=%s  total_elapsed=%s  budget_left=%s",
            config_id, seed, epoch + 1, train_cfg.epochs,
            train_loss, val_loss, best_val_loss, best_epoch + 1,
            lr_now, patience_counter, train_cfg.patience,
            skipped_batches_epoch,
            time.strftime("%H:%M:%S", time.gmtime(elapsed)),
            time.strftime("%H:%M:%S", time.gmtime(total_elapsed)),
            time.strftime("%H:%M:%S", time.gmtime(budget_left)) if np.isfinite(budget_left) else "n/a",
        )

        if writer:
            writer.add_scalar(f"Loss/train", train_loss, epoch)
            writer.add_scalar(f"Loss/val",   val_loss,   epoch)
            writer.add_scalar(f"LR",         lr_now,     epoch)

        if patience_counter >= train_cfg.patience:
            log.info(
                "[%s|seed=%d] Early stopping at epoch %d (patience=%d).",
                config_id, seed, epoch + 1, train_cfg.patience,
            )
            break

    if writer:
        writer.close()

    final_ckpt_path = ckpt_dir / f"{config_id}_seed{seed}_last.pt"
    final_payload = {
        "epoch":        len(train_loss_hist) - 1,
        "model_state":  model.state_dict(),
        "optim_state":  optimiser.state_dict(),
        "val_loss":     val_loss_hist[-1] if val_loss_hist else float("inf"),
        "config_id":    config_id,
        "seed":         seed,
        "model_config": model.cfg,
    }
    torch.save(final_payload, final_ckpt_path)
    if not ckpt_path.exists():
        log.warning(
            "[%s|seed=%d] best-checkpoint file was missing at end of training; promoting last checkpoint to %s",
            config_id, seed, ckpt_path,
        )
        torch.save(final_payload, ckpt_path)

    return {
        "train_loss_history": train_loss_hist,
        "val_loss_history":   val_loss_hist,
        "best_val_loss":      best_val_loss,
        "best_epoch":         best_epoch,
        "checkpoint_path":    str(ckpt_path),
        "final_checkpoint_path": str(final_ckpt_path),
        "skipped_batches":    skipped_batches_total,
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Validation loss helper
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@torch.no_grad()
def _evaluate_loss(
    model:      BaseModel,
    loader:     DataLoader,
    train_cfg:  TrainConfig,
    device:     str,
    beta:       float = 1.0,
    pos_weight: Optional[torch.Tensor] = None,
) -> dict:
    model.eval()
    use_amp = torch.cuda.is_available() and str(device).startswith("cuda") if train_cfg.use_amp is None else bool(train_cfg.use_amp)
    total_losses: list[float] = []
    primary_losses: list[float] = []
    recon_losses: list[float] = []
    kl_losses: list[float] = []
    for batch in loader:
        batch = _sanitize_batch(_to_device(batch, device))
        with autocast(device_type="cuda", enabled=use_amp):
            if getattr(model.cfg, "use_generative", False):
                outputs = _safe_forward(model, batch, return_cvae=True, context="validation")
                if outputs is None:
                    continue
                mu, log_var, cvae_out = outputs
                log_var = torch.clamp(log_var, min=-10.0, max=10.0)
                gnll_loss = regression_loss(
                    mu, log_var, batch["pIC50"],
                    weight=None,   # no weighting during validation
                )
                if "target_seq" in batch and batch["target_seq"] is not None:
                    input_seq = batch["target_seq"][:, :-1]
                    target_out = batch["target_seq"][:, 1:]
                    elbo, recon, kl = cvae_elbo_loss(
                        cvae_out["logits"],
                        target_out,
                        cvae_out["mu_z"],
                        cvae_out["log_var_z"],
                        beta=beta,
                        recon_weight=train_cfg.cvae_recon_weight,
                    )
                    total_loss = gnll_loss + elbo
                    recon_losses.append(recon.item())
                    kl_losses.append(kl.item())
                else:
                    total_loss = gnll_loss
                    recon_losses.append(0.0)
                    kl_losses.append(0.0)
                primary_loss = gnll_loss
            elif model.cfg.task_type == "classification":
                logits = _safe_forward(model, batch, context="validation")
                if logits is None:
                    continue
                labels = _labels_from_batch(batch, model.cfg, train_cfg)
                primary_loss = classification_loss(
                    logits,
                    labels,
                    pos_weight=pos_weight,
                )
                total_loss = primary_loss
                recon_losses.append(0.0)
                kl_losses.append(0.0)
            else:
                outputs = _safe_forward(model, batch, context="validation")
                if outputs is None:
                    continue
                mu, log_var = outputs
                log_var = torch.clamp(log_var, min=-10.0, max=10.0)
                primary_loss = regression_loss(
                    mu, log_var, batch["pIC50"],
                    weight=None,   # no weighting during validation
                )
                total_loss = primary_loss
                recon_losses.append(0.0)
                kl_losses.append(0.0)
        total_losses.append(total_loss.item())
        primary_losses.append(primary_loss.item())
    return {
        "total_loss": float(np.mean(total_losses)) if total_losses else float("inf"),
        "primary_loss": float(np.mean(primary_losses)) if primary_losses else float("inf"),
        "recon_loss": float(np.mean(recon_losses)) if recon_losses else 0.0,
        "kl_loss": float(np.mean(kl_losses)) if kl_losses else 0.0,
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Device transfer for batch dict
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _to_device(batch: dict, device: str) -> dict:
    """Move all tensor values in a batch dict to device (in-place)."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _sanitize_batch(batch: dict) -> dict:
    """Clamp non-finite values before the model sees them."""
    sanitized = dict(batch)
    for key in ("physchem", "morgan_fp", "esm_pocket", "confidence", "pIC50", "pIC50_std", "ic50_nm", "label", "weight"):
        value = sanitized.get(key)
        if isinstance(value, torch.Tensor):
            sanitized[key] = torch.nan_to_num(value, nan=0.0, posinf=1e4, neginf=-1e4)
    return sanitized


def _safe_forward(
    model: BaseModel,
    batch: dict,
    return_cvae: bool = False,
    context: str = "forward",
):
    """
    Guard model forward passes so tensor-shape and missing-field errors do not
    crash the whole run. The caller decides whether to skip the batch.
    """
    try:
        if return_cvae:
            return model(batch, return_cvae=True)
        return model(batch)
    except Exception as exc:
        log.error("Model forward failed during %s: %s", context, exc, exc_info=True)
        return None


def load_saved_split_indices(
    config_id: Optional[str] = None,
    seed: int = 42,
    checkpoint_dir: str = "./checkpoints",
) -> list[int]:
    """
    Load saved test split indices for a given seed.

    The split file is shared across configs, so *config_id* is accepted only
    for interface compatibility with downstream modules.
    """
    candidate_paths = [
        Path(checkpoint_dir) / f"split_indices_{config_id}_seed{seed}.json" if config_id else None,
        Path(checkpoint_dir) / f"split_indices_seed{seed}.json",
        Path(f"./split_indices_seed{seed}.json"),
    ]
    for path in candidate_paths:
        if path is None:
            continue
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                split_indices = json.load(f)
            test_idx = split_indices.get("test_idx")
            if test_idx is None:
                raise KeyError(f"'test_idx' missing from split file: {path}")
            return test_idx
    raise FileNotFoundError(
        f"Could not find split indices for seed={seed} in {candidate_paths}"
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Multi-seed training for a single configuration
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def train_config(
    config_id:     str,
    dataset:       KinaseLigandDataset,
    train_cfg:     TrainConfig,
    device:        str = DEVICE,
) -> dict:
    """
    Train *config_id* across all seeds.

    Returns a dict:
        {seed: run_result_dict}
    where run_result_dict has keys from train_one_seed().
    """
    log.info("=" * 65)
    log.info("Training configuration: %s", config_id)
    log.info("Seeds: %s", train_cfg.seeds)
    log.info("=" * 65)

    model_cfg = _effective_model_config(config_id, train_cfg)
    eligible_indices = _eligible_indices_for_config(dataset, model_cfg, train_cfg)
    if not eligible_indices:
        raise RuntimeError(f"No eligible samples for config '{config_id}'.")

    # Build splits once (scaffold split uses smiles, which is seed-agnostic
    # for scaffold assignment, but the assignment of scaffolds to splits uses
    # the first seed to ensure reproducibility across configs)
    subset_df = dataset.df.iloc[eligible_indices].reset_index(drop=True)
    smiles_list = subset_df["smiles"].tolist()
    if train_cfg.split_strategy == "scaffold":
        train_pos, val_pos, test_pos = murcko_scaffold_split(
            smiles_list,
            val_frac   = train_cfg.val_fraction,
            test_frac  = train_cfg.test_fraction,
            seed       = train_cfg.seeds[0],
        )
    else:
        train_pos, val_pos, test_pos = random_split(
            len(eligible_indices),
            val_frac   = train_cfg.val_fraction,
            test_frac  = train_cfg.test_fraction,
            seed       = train_cfg.seeds[0],
        )

    train_idx = [eligible_indices[i] for i in train_pos]
    val_idx = [eligible_indices[i] for i in val_pos]
    test_idx = [eligible_indices[i] for i in test_pos]

    # Save split indices for evaluation
    split_indices = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
    split_path = Path(train_cfg.checkpoint_dir) / f"split_indices_{config_id}_seed{train_cfg.seeds[0]}.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_indices, f)

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)
    pos_weight = _compute_pos_weight(dataset, train_idx, model_cfg, train_cfg, device)

    train_loader_kwargs = _build_loader_kwargs(train_cfg, is_train=True)
    val_loader_kwargs = _build_loader_kwargs(train_cfg, is_train=False)

    log.info(
        "DataLoader settings | train_workers=%d val_workers=%d pin_memory=%s persistent_workers=%s prefetch_factor=%s amp=%s max_batch_weight=%.2f",
        train_loader_kwargs["num_workers"],
        val_loader_kwargs["num_workers"],
        train_loader_kwargs["pin_memory"],
        train_loader_kwargs.get("persistent_workers", False),
        train_loader_kwargs.get("prefetch_factor", "n/a"),
        torch.cuda.is_available() if train_cfg.use_amp is None else train_cfg.use_amp,
        train_cfg.max_batch_weight,
    )

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size,
        shuffle=True, collate_fn=collate_fn,
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size * 2,
        shuffle=False, collate_fn=collate_fn,
        **val_loader_kwargs,
    )

    seed_results: dict[int, dict] = {}

    for seed in progress_iter(
        train_cfg.seeds,
        total=len(train_cfg.seeds),
        desc=f"{config_id} seeds",
        leave=True,
    ):
        log.info("-" * 55)
        log.info("Seed %d | config=%s", seed, config_id)
        log.info("-" * 55)

        # Re-instantiate model for each seed (fresh weights)
        set_seed(seed)
        model = BaseModel(copy.deepcopy(model_cfg))

        result = train_one_seed(
            model       = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            train_cfg    = train_cfg,
            seed         = seed,
            config_id    = config_id,
            pos_weight   = pos_weight,
            device       = device,
        )
        seed_results[seed] = result
        seed_results[seed]["test_indices"] = test_idx

    return seed_results


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Full experiment: train all configurations
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def train_all_configs(
    dataset:    KinaseLigandDataset,
    train_cfg:  TrainConfig,
    config_ids: Optional[list[str]] = None,
    device:     str = DEVICE,
) -> dict:
    """
    Train all (or a subset of) configurations across all seeds.

    Returns
    -------
    all_results : {config_id: {seed: run_result_dict}}
    """
    if config_ids is None:
        config_ids = ALL_CONFIG_IDS

    all_results: dict[str, dict] = {}
    for cid in progress_iter(
        config_ids,
        total=len(config_ids),
        desc="Model configs",
        leave=True,
    ):
        try:
            all_results[cid] = train_config(cid, dataset, train_cfg, device)
        except Exception as exc:
            log.error(
                "Training failed for config '%s': %s", cid, exc, exc_info=True
            )
            all_results[cid] = {"error": str(exc)}

    return all_results


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Checkpoint loading
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def load_checkpoint(
    config_id: str,
    seed:      int,
    checkpoint_dir: str = "./checkpoints",
    device:    str = DEVICE,
) -> tuple[BaseModel, dict]:
    """
    Load the best checkpoint for (config_id, seed).

    Returns
    -------
    (model, checkpoint_dict)
    """
    ckpt_path = Path(checkpoint_dir) / f"{config_id}_seed{seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg: ModelConfig = ckpt["model_config"]
    model = BaseModel(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    log.info(
        "Loaded checkpoint: %s (epoch=%d, val_loss=%.4f)",
        ckpt_path, ckpt["epoch"], ckpt["val_loss"],
    )
    return model, ckpt


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Prediction helper (used by module7 and module8)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@torch.no_grad()
def predict(
    model:   BaseModel,
    loader:  DataLoader,
    device:  str = DEVICE,
) -> dict:
    """
    Run inference on a DataLoader.

    Returns
    -------
    dict with:
        mu / logits : np.ndarray (N,)
        var         : np.ndarray (N,) for regression only
        probs       : np.ndarray (N,) for classification only
        targets  : np.ndarray (N,)   ground-truth pIC50
        labels   : np.ndarray (N,)   classification labels when relevant
        weights  : np.ndarray (N,)   sample weights
        inchikeys: list[str]
        uniprots : list[str]
    """
    model.eval()
    all_mu, all_var, all_probs, all_labels, all_tgt, all_wt = [], [], [], [], [], []
    all_ik, all_uid = [], []

    for batch in loader:
        batch = _to_device(batch, device)
        if model.cfg.task_type == "classification":
            logits = _safe_forward(model, batch, context="predict")
            if logits is None:
                log.warning("Skipping prediction batch after forward failure.")
                continue
            probs = torch.sigmoid(logits.squeeze(-1))
            labels = _labels_from_batch(batch, model.cfg, TrainConfig())
            all_mu.append(logits.squeeze(-1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        else:
            if getattr(model.cfg, "use_generative", False):
                outputs = _safe_forward(model, batch, context="predict")
                if outputs is None:
                    log.warning("Skipping prediction batch after forward failure.")
                    continue
                mu, log_var = outputs
                log_var = torch.clamp(log_var, min=-10.0, max=10.0)
            else:
                outputs = _safe_forward(model, batch, context="predict")
                if outputs is None:
                    log.warning("Skipping prediction batch after forward failure.")
                    continue
                mu, log_var = outputs
                log_var = torch.clamp(log_var, min=-10.0, max=10.0)

            all_mu.append(mu.squeeze(-1).cpu().numpy())
            all_var.append(torch.exp(log_var.squeeze(-1)).cpu().numpy())
        all_tgt.append(batch["pIC50"].cpu().numpy())
        all_wt.append(batch["weight"].cpu().numpy())
        all_ik.extend(batch["inchikey"])
        all_uid.extend(batch["uniprot_id"])

    if not all_mu:
        raise RuntimeError("Prediction produced no valid batches. Check logs for earlier forward failures.")

    out = {
        "mu":        np.concatenate(all_mu),
        "targets":   np.concatenate(all_tgt),
        "weights":   np.concatenate(all_wt),
        "inchikeys": all_ik,
        "uniprots":  all_uid,
    }
    if all_var:
        out["var"] = np.concatenate(all_var)
    if all_probs:
        out["probs"] = np.concatenate(all_probs)
        out["labels"] = np.concatenate(all_labels)
    return out


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Entry point
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    import argparse
    from module2_feature_engineering import load_ligand_feature_store
    from module3_protein_features import load_protein_feature_store

    parser = argparse.ArgumentParser(description="Train kinaseâ€“ligand models.")
    parser.add_argument("--dataset",    default="./pipeline_outputs/dataset_clean.parquet")
    parser.add_argument("--lig-store",  default="./pipeline_outputs/ligand_features")
    parser.add_argument("--prot-store", default="./pipeline_outputs/protein_feature_store.pt")
    parser.add_argument("--config",     default="full_model",
                        choices=ALL_CONFIG_IDS + ["all"])
    parser.add_argument("--seeds",      nargs="+", type=int, default=[42])
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64 if torch.cuda.is_available() else 24)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--active-cutoff-nm", type=float, default=500.0)
    parser.add_argument("--inactive-cutoff-nm", type=float, default=5000.0)
    parser.add_argument("--binary-threshold-pic50", type=float, default=7.0)
    parser.add_argument("--min-weight-std", type=float, default=0.05)
    parser.add_argument("--min-protein-pocket-confidence", type=float, default=0.5)
    parser.add_argument("--max-batch-weight", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device",     default=DEVICE)
    parser.add_argument("--no-scaffold", action="store_true")
    args = parser.parse_args()

    df           = pd.read_parquet(args.dataset)
    lig_store    = load_ligand_feature_store(args.lig_store)
    prot_store   = load_protein_feature_store(args.prot_store)

    dataset = KinaseLigandDataset(
        df,
        lig_store,
        prot_store,
        min_weight_std=args.min_weight_std,
        min_protein_pocket_confidence=args.min_protein_pocket_confidence,
    )

    train_cfg = TrainConfig(
        seeds           = args.seeds,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        lr              = args.lr,
        active_cutoff_nm = args.active_cutoff_nm,
        inactive_cutoff_nm = args.inactive_cutoff_nm,
        binary_threshold_pic50 = args.binary_threshold_pic50,
        min_weight_std  = args.min_weight_std,
        min_protein_pocket_confidence = args.min_protein_pocket_confidence,
        max_batch_weight = args.max_batch_weight,
        num_workers     = args.num_workers,
        prefetch_factor = args.prefetch_factor,
        persistent_workers = not args.no_persistent_workers,
        pin_memory      = False if args.no_pin_memory else None,
        use_amp         = False if args.no_amp else None,
        split_strategy  = "random" if args.no_scaffold else "scaffold",
    )

    config_ids = ALL_CONFIG_IDS if args.config == "all" else [args.config]
    all_results = train_all_configs(dataset, train_cfg, config_ids, device=args.device)
    log.info("Training complete. Checkpoints in: %s", train_cfg.checkpoint_dir)

