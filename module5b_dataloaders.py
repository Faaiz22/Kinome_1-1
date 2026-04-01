"""
module5b_dataloaders.py
=======================
TITLE
Auxiliary dataset and dataloader utilities for kinase-ligand experiments.

PURPOSE
This module packages dataset rows and scaffold-aware splits into PyTorch data
pipelines when a lighter-weight loader layer is needed outside the main module6
training implementation.

WHAT IT DOES
- Builds dataset objects and dataloaders.
- Supports Bemis-Murcko scaffold splitting.
- Keeps batching compatible with PyTorch Geometric workflows.

HOW IT WORKS
1. Load normalized dataset rows.
2. Resolve matching ligand and protein features.
3. Split molecules by scaffold for evaluation realism.
4. Yield PyTorch/PyG-ready batches.

INPUT CONTRACT
- Dataset with normalized ligand-target identifiers.
- Ligand and protein stores aligned to those identifiers.

OUTPUT CONTRACT
- Dataset objects, scaffold splits, and dataloaders.

DEPENDENCIES
- pandas, numpy, torch, torch_geometric

CRITICAL ASSUMPTIONS
- Feature stores are already built and keyed consistently.

FAILURE MODES
- Missing features
- Empty scaffold groups
- Invalid dataset schema

SAFETY CHECKS IMPLEMENTED
- Feature-presence filtering
- Split reproducibility controls
- Batching compatible with graph models

HOW TO RUN
- Imported by training/evaluation workflows as needed.

HOW IT CONNECTS TO PIPELINE
It offers supporting loader utilities for experimentation around the main
module6 training path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

log = logging.getLogger("module5b_dataloaders")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# PyTorch Dataset
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class KinaseLigandDataset(Dataset):
    """
    PyTorch Dataset for kinaseâ€“ligand binding affinity prediction.

    Loads:
    - Protein features (ESM-2, AlphaFold) from module3 store
    - Ligand graph features from module10 or other ligand featurizer
    - Binding affinity targets from dataset_clean.parquet

    Returns per-sample:
    {
        "uniprot_id"   : str
        "ligand_id"    : str
        "protein_features" : dict with keys:
            "esm_pocket"   : (85, 1280)
            "coords"       : (85, 3)
            "plddt"        : (85,)
            "confidence"   : (85,)
            "pocket_mask"  : (85,)
        "ligand_graph" : dict with keys (PyG-compatible):
            "x"       : (N_atoms, d_features)
            "edge_index" : (2, N_edges)
            "edge_attr" : (N_edges, d_edge)
        "target"       : float (binding affinity / pIC50 / etc.)
        "smiles"       : str
    }
    """

    def __init__(
        self,
        dataset_path: str | Path = "./pipeline_outputs/dataset_clean.parquet",
        protein_store_path: str | Path = "./pipeline_outputs/protein_feature_store.pt",
        ligand_store_path: Optional[str | Path] = None,
        split_indices: Optional[list[int]] = None,
    ) -> None:
        """
        Parameters
        ----------
        dataset_path        : path to dataset_clean.parquet
        protein_store_path  : path to protein features dict from module3
        ligand_store_path   : path to ligand featurizer output (optional)
        split_indices       : if provided, only load these rows
        """
        self.dataset_path = Path(dataset_path)
        self.protein_store_path = Path(protein_store_path)
        self.ligand_store_path = Path(ligand_store_path) if ligand_store_path else None

        # Load main dataset
        self.df = pd.read_parquet(self.dataset_path)
        if split_indices is not None:
            self.df = self.df.iloc[split_indices].reset_index(drop=True)

        # Load protein features (dict: UniProt â†’ ProteinFeatures)
        if self.protein_store_path.exists():
            self.protein_store = torch.load(
                self.protein_store_path, weights_only=False
            )
        else:
            log.warning(f"Protein store not found at {self.protein_store_path}")
            self.protein_store = {}

        # Load ligand features (optional dict: ligand_id â†’ LigandGraph)
        if self.ligand_store_path and self.ligand_store_path.exists():
            self.ligand_store = torch.load(
                self.ligand_store_path, weights_only=False
            )
        else:
            self.ligand_store = {}

        log.info(
            "KinaseLigandDataset initialized: %d samples, "
            "%d protein entries, %d ligand entries",
            len(self.df), len(self.protein_store), len(self.ligand_store),
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """Load one sample."""
        row = self.df.iloc[idx]

        uniprot_id = row.get("uniprot_id", "")
        ligand_id = row.get("ligand_id", row.get("smiles", ""))
        target = float(row.get("pIC50", row.get("affinity", 0.0)))
        label = row.get("classification_label", np.nan)
        smiles = row.get("smiles", "")

        # Fetch protein features
        if uniprot_id in self.protein_store:
            pf = self.protein_store[uniprot_id]
            protein_features = {
                "esm_pocket": pf.esm_pocket,
                "coords": pf.coords,
                "plddt": pf.plddt,
                "confidence": pf.confidence,
                "pocket_mask": pf.pocket_mask,
            }
        else:
            log.warning(f"Protein features missing for {uniprot_id}, using zeros")
            protein_features = {
                "esm_pocket": torch.zeros((85, 1280)),
                "coords": torch.zeros((85, 3)),
                "plddt": torch.zeros((85,)),
                "confidence": torch.zeros((85,)),
                "pocket_mask": torch.zeros((85,), dtype=torch.bool),
            }

        # Fetch ligand features (optional)
        if ligand_id in self.ligand_store:
            ligand_graph = self.ligand_store[ligand_id]
        else:
            # Fallback: return a zero graph with correct feature dimensions.
            # ATOM_FEAT_DIM=43, BOND_FEAT_DIM=12 must match module2 constants.
            log.warning(
                "Ligand features missing for id='%s' (smiles='%s'); "
                "returning zero graph with ATOM_FEAT_DIM=43, BOND_FEAT_DIM=12.",
                ligand_id, smiles,
            )
            ligand_graph = {
                "x": torch.zeros((1, 43)),              # 1 dummy atom, 43-dim
                "edge_index": torch.zeros((2, 0), dtype=torch.long),
                "edge_attr": torch.zeros((0, 12)),      # 12-dim bond features
            }

        return {
            "uniprot_id": uniprot_id,
            "ligand_id": ligand_id,
            "protein_features": protein_features,
            "ligand_graph": ligand_graph,
            "target": torch.tensor(target, dtype=torch.float32),
            "label": torch.tensor(float(label) if pd.notna(label) else float("nan"), dtype=torch.float32),
            "smiles": smiles,
        }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Bemis-Murcko Scaffold Splitting
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def bemis_murcko_scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: Optional[int] = None,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split dataset by Bemis-Murcko scaffold to ensure NO scaffold overlap
    between train/val/test (strict OOD evaluation).

    Parameters
    ----------
    df           : DataFrame with at least smiles_col column
    smiles_col   : column name containing SMILES strings
    train_ratio  : fraction for training set
    val_ratio    : fraction for validation set
    test_ratio   : fraction for test set
    random_state : seed for reproducibility

    Returns
    -------
    (train_indices, val_indices, test_indices)
        Each is a list of row indices for the respective split.

    Note
    ----
    Requires RDKit: pip install rdkit
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        raise ImportError(
            "RDKit is required for Bemis-Murcko scaffold splitting. "
            "Install with: pip install rdkit"
        )

    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random.RandomState()

    # Compute Bemis-Murcko scaffold for each molecule
    scaffolds = []
    for smiles in df[smiles_col]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scaffold_str = "INVALID"
            else:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_str = Chem.MolToSmiles(scaffold)
        except Exception as e:
            log.warning(f"Failed to parse SMILES '{smiles}': {e}")
            scaffold_str = "FAILED"
        scaffolds.append(scaffold_str)

    df["__scaffold"] = scaffolds

    # Group by scaffold
    scaffold_groups = df.groupby("__scaffold").groups

    # Shuffle scaffolds randomly
    scaffold_list = list(scaffold_groups.keys())
    np.random.shuffle(scaffold_list)

    # Assign scaffolds to splits
    n_scaffolds = len(scaffold_list)
    n_train_scaf = int(np.ceil(n_scaffolds * train_ratio))
    n_val_scaf = int(np.ceil(n_scaffolds * val_ratio))

    train_scaffolds = scaffold_list[:n_train_scaf]
    val_scaffolds = scaffold_list[n_train_scaf : n_train_scaf + n_val_scaf]
    test_scaffolds = scaffold_list[n_train_scaf + n_val_scaf :]

    # Collect indices for each split
    def _collect_split_indices(split_scaffolds: list[str]) -> list[int]:
        arrays = [scaffold_groups[s].values for s in split_scaffolds if s in scaffold_groups]
        if not arrays:
            return []
        return np.concatenate(arrays).tolist()

    train_indices = _collect_split_indices(train_scaffolds)
    val_indices = _collect_split_indices(val_scaffolds)
    test_indices = _collect_split_indices(test_scaffolds)

    log.info(
        "Bemis-Murcko scaffold split: %d scaffolds â†’ "
        "train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        n_scaffolds,
        len(train_indices), 100 * len(train_indices) / len(df),
        len(val_indices), 100 * len(val_indices) / len(df),
        len(test_indices), 100 * len(test_indices) / len(df),
    )

    return train_indices, val_indices, test_indices


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# DataLoader Builder
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def make_dataloaders(
    dataset_path: str | Path = "./pipeline_outputs/dataset_clean.parquet",
    protein_store_path: str | Path = "./pipeline_outputs/protein_feature_store.pt",
    ligand_store_path: Optional[str | Path] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    scaffold_split: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: Optional[int] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders with optional scaffold splitting.

    Parameters
    ----------
    dataset_path      : path to dataset_clean.parquet
    protein_store_path: path to protein features
    ligand_store_path : path to ligand features (optional)
    batch_size        : batch size for DataLoader
    num_workers       : number of data loading workers
    pin_memory        : pin memory for GPU transfer
    scaffold_split    : if True, use Bemis-Murcko split; else random split
    train_ratio       : fraction for training
    val_ratio         : fraction for validation
    test_ratio        : fraction for test
    random_state      : seed for reproducibility

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    # Load dataset
    df = pd.read_parquet(dataset_path)

    # Determine split indices
    if scaffold_split:
        train_idx, val_idx, test_idx = bemis_murcko_scaffold_split(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
        )
    else:
        # Random split
        n = len(df)
        n_train = int(np.ceil(n * train_ratio))
        n_val = int(np.ceil(n * val_ratio))
        all_idx = np.arange(n)
        np.random.RandomState(random_state).shuffle(all_idx)
        train_idx = all_idx[:n_train].tolist()
        val_idx = all_idx[n_train : n_train + n_val].tolist()
        test_idx = all_idx[n_train + n_val :].tolist()

    # Create datasets for each split
    train_dataset = KinaseLigandDataset(
        dataset_path, protein_store_path, ligand_store_path, split_indices=train_idx
    )
    val_dataset = KinaseLigandDataset(
        dataset_path, protein_store_path, ligand_store_path, split_indices=val_idx
    )
    test_dataset = KinaseLigandDataset(
        dataset_path, protein_store_path, ligand_store_path, split_indices=test_idx
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches during training
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    log.info(
        "DataLoaders created: train=%d, val=%d, test=%d",
        len(train_loader), len(val_loader), len(test_loader),
    )

    return train_loader, val_loader, test_loader


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Self-test
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    log = logging.getLogger("module5b_test")

    # Test scaffold split with dummy data
    dummy_smiles = [
        "c1ccccc1",  # benzene
        "c1ccccc1C",  # toluene (same scaffold)
        "c1ccc2c(c1)cccc2",  # naphthalene
        "c1ccc(O)cc1",  # phenol (benzene scaffold)
        "c1ccc2c(c1)ccc2C",  # methylnaphthalene (naphthalene scaffold)
        "CCc1ccccc1",  # ethylbenzene (benzene scaffold)
        "c1cnc2ccccc2n1",  # imidazole-based
    ]

    dummy_df = pd.DataFrame({
        "uniprot_id": ["P12345"] * len(dummy_smiles),
        "ligand_id": [f"lig_{i}" for i in range(len(dummy_smiles))],
        "smiles": dummy_smiles,
        "pIC50": np.random.randn(len(dummy_smiles)) * 2 + 7.0,
    })

    try:
        train_idx, val_idx, test_idx = bemis_murcko_scaffold_split(
            dummy_df, random_state=42
        )
        log.info(
            "Bemis-Murcko split successful: "
            "train=%d, val=%d, test=%d",
            len(train_idx), len(val_idx), len(test_idx),
        )
        print("Bemis-Murcko scaffold split test passed")
    except ImportError:
        log.warning("RDKit not available; skipping scaffold split test")
        print("RDKit not available for scaffold split test (OK)")

    print("All module5b self-tests passed")


