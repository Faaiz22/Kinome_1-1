"""
module2_feature_engineering.py
==============================
TITLE
Ligand feature engineering for kinase-ligand modelling.

PURPOSE
This module converts standardized SMILES into reproducible ligand features that
training, evaluation, and inference can reuse without changing scientific
meaning between stages.

WHAT IT DOES
- Builds graph features, physicochemical descriptors, Morgan fingerprints, and
  optional 3D conformers.
- Saves and reloads a ligand feature store keyed by InChIKey.
- Supports cache-aware bulk feature generation.

HOW IT WORKS
1. Parse standardized SMILES with RDKit.
2. Build atom and bond feature tensors.
3. Compute descriptor vectors and optional conformers.
4. Package features in a PyG-compatible `Data` object.
5. Persist one feature object per InChIKey.

INPUT CONTRACT
- Valid standardized SMILES strings.
- Clean dataset rows containing `smiles` and `inchikey` for bulk build.

OUTPUT CONTRACT
- `Data` objects containing graph, physchem, fingerprint, and identifier fields.
- Disk-backed or legacy ligand feature stores.

DEPENDENCIES
- RDKit, torch, torch_geometric, pandas, numpy

CRITICAL ASSUMPTIONS
- SMILES are already standardized by upstream code.
- InChIKey is the stable ligand cache key across the pipeline.

FAILURE MODES
- Invalid SMILES
- Molecules with zero heavy atoms
- 3D conformer generation failure

SAFETY CHECKS IMPLEMENTED
- RDKit parse validation
- NaN protection for descriptor tensors
- Cache-aware safe loading and resumable store building

HOW TO RUN
- `python module2_feature_engineering.py --dataset ./pipeline_outputs/dataset_clean.parquet --output ./pipeline_outputs/ligand_features`

HOW IT CONNECTS TO PIPELINE
+Used by dataset preparation, training, uncertainty analysis, and Streamlit
+inference so ligand featurization stays identical everywhere.
"""

from __future__ import annotations

import logging
import errno
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Mapping, Iterable

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, GraphDescriptors, rdMolDescriptors
from rdkit.Chem.rdchem import HybridizationType, ChiralType, BondStereo, BondType

try:
    from torch_geometric.data import Data
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    # Minimal stand-in so the module is importable even without torch_geometric
    class Data:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

log = logging.getLogger("module2")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Vocabulary tables for one-hot encoding
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ATOM_ELEMENTS = [
    "C", "N", "O", "S", "F", "Cl", "Br", "I",
    "P", "Si", "B", "Se", "Te", "<other>",
]

ATOM_DEGREES    = [0, 1, 2, 3, 4, 5, 6, "<other>"]
ATOM_FCHARGE    = [-3, -2, -1, 0, 1, 2, 3, "<other>"]
ATOM_HYBRID     = [
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
    HybridizationType.OTHER,
]
ATOM_CHIRAL     = [
    ChiralType.CHI_UNSPECIFIED,
    ChiralType.CHI_TETRAHEDRAL_CW,
    ChiralType.CHI_TETRAHEDRAL_CCW,
    ChiralType.CHI_OTHER,
]

BOND_TYPES      = [
    BondType.SINGLE,
    BondType.DOUBLE,
    BondType.TRIPLE,
    BondType.AROMATIC,
    "<other>",
]
BOND_STEREO     = [
    BondStereo.STEREONONE,
    BondStereo.STEREOANY,
    BondStereo.STEREOE,
    BondStereo.STEREOZ,
    "<other>",
]

# Atom feature dimensionality:
# element(14) + degree(8) + fcharge(8) + hybrid(6) + chiral(4) +
# is_aromatic(1) + is_in_ring(1) + num_Hs(1) = 43 dims total
# After collapsing num_Hs to an integer clipped to [0,4] it stays
# at 43 (we embed it as a raw int, see below).
ATOM_FEAT_DIM: int = (
    len(ATOM_ELEMENTS) +   # 14
    len(ATOM_DEGREES)   +  #  8
    len(ATOM_FCHARGE)   +  #  8
    len(ATOM_HYBRID)    +  #  6
    len(ATOM_CHIRAL)    +  #  4
    1                   +  # is_aromatic
    1                   +  # is_in_ring
    1                      # num_implicit_Hs (int, clipped 0-4)
)  # = 43

# Bond feature dimensionality:
# bond_type(5) + is_conjugated(1) + is_in_ring(1) + bond_stereo(5) = 12
BOND_FEAT_DIM: int = (
    len(BOND_TYPES)   +   # 5
    1                 +   # is_conjugated
    1                 +   # is_in_ring
    len(BOND_STEREO)      # 5
)  # = 12


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# One-hot helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _one_hot(value, vocab: list) -> list[float]:
    """One-hot encode *value* against *vocab*. Falls back to last element."""
    if value not in vocab:
        value = vocab[-1]   # "<other>" sentinel
    return [float(value == v) for v in vocab]


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Atom featuriser
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def atom_features(atom: Chem.rdchem.Atom) -> list[float]:
    """
    Return a flat float list of length ATOM_FEAT_DIM for one RDKit atom.

    Features
    --------
    - Element symbol  (14-way one-hot)
    - Degree          (8-way one-hot)
    - Formal charge   (8-way one-hot)
    - Hybridisation   (6-way one-hot)
    - Chirality       (4-way one-hot)
    - is_aromatic     (binary)
    - is_in_ring      (binary)
    - num_implicit_Hs (integer, clipped 0-4, normalised by /4)
    """
    return (
        _one_hot(atom.GetSymbol(),             ATOM_ELEMENTS)
        + _one_hot(atom.GetDegree(),           ATOM_DEGREES)
        + _one_hot(atom.GetFormalCharge(),     ATOM_FCHARGE)
        + _one_hot(atom.GetHybridization(),    ATOM_HYBRID)
        + _one_hot(atom.GetChiralTag(),        ATOM_CHIRAL)
        + [float(atom.GetIsAromatic())]
        + [float(atom.IsInRing())]
        + [min(atom.GetTotalNumHs(), 4) / 4.0]
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Bond featuriser
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def bond_features(bond: Chem.rdchem.Bond) -> list[float]:
    """
    Return a flat float list of length BOND_FEAT_DIM for one RDKit bond.

    Features
    --------
    - Bond type       (5-way one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC, other)
    - is_conjugated   (binary)
    - is_in_ring      (binary)
    - Bond stereo     (5-way one-hot)
    """
    return (
        _one_hot(bond.GetBondType(),         BOND_TYPES)
        + [float(bond.GetIsConjugated())]
        + [float(bond.IsInRing())]
        + _one_hot(bond.GetStereo(),         BOND_STEREO)
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Graph builder (mol â†’ PyG Data)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def mol_to_graph(
    mol: Chem.rdchem.Mol,
    inchikey: str = "",
) -> Data:
    """
    Convert an RDKit molecule to a PyTorch Geometric Data object.

    Parameters
    ----------
    mol      : rdkit.Chem.rdchem.Mol (with explicit Hs removed for GNN use)
    inchikey : optional identifier stored on the Data object

    Returns
    -------
    torch_geometric.data.Data with:
        x          : float tensor (N_atoms, ATOM_FEAT_DIM)
        edge_index : long tensor  (2, 2 * N_bonds)   â€” bidirectional
        edge_attr  : float tensor (2 * N_bonds, BOND_FEAT_DIM)
        inchikey   : str
        num_atoms  : int
    """
    # Atom features
    atom_feat_list = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(atom_feat_list, dtype=torch.float)

    # Bond features â€” edges are added in BOTH directions (iâ†’j and jâ†’i)
    src_list, dst_list, bond_feat_list = [], [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        src_list += [i, j]
        dst_list += [j, i]
        bond_feat_list += [bf, bf]   # same features for both directions

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.tensor(bond_feat_list, dtype=torch.float)
    else:
        # Isolated atom (e.g. noble gas ligand â€” rare but defensive)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        inchikey=inchikey,
        num_atoms=mol.GetNumAtoms(),
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Physicochemical feature vector
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PHYSCHEM_NAMES: list[str] = [
    "MolWt", "ExactMolWt", "MolLogP", "TPSA",
    "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "NumAromaticRings", "NumAliphaticRings", "NumHeavyAtoms",
    "NumSaturatedRings", "FractionCSP3", "LabuteASA",
    "BalabanJ", "BertzCT",
    "Chi0n", "Chi1n", "Kappa1", "Kappa2",
    "RingCount", "NumRadicalElectrons", "MaxPartialCharge",
]
PHYSCHEM_DIM: int = len(PHYSCHEM_NAMES)

# Morgan fingerprint (ECFP4) dimension â€” stored separately on the Data object
# as `data.morgan_fp` so the GNN path is unaffected.
MORGAN_RADIUS: int = 2
MORGAN_NBITS:  int = 1024


def _safe_descriptor(fn, mol) -> float:
    """Compute one RDKit descriptor, returning NaN on failure."""
    try:
        val = fn(mol)
        return float(val) if val is not None else float("nan")
    except Exception:
        return float("nan")


def compute_physchem(mol: Chem.rdchem.Mol) -> np.ndarray:
    """
    Compute the physicochemical feature vector for a molecule.

    Returns
    -------
    np.ndarray of shape (PHYSCHEM_DIM,)  dtype=float32
    NaN values are returned for any descriptor that fails to compute;
    the caller is responsible for imputation if needed.
    """
    fns = [
        Descriptors.MolWt,
        Descriptors.ExactMolWt,
        Descriptors.MolLogP,
        Descriptors.TPSA,
        rdMolDescriptors.CalcNumHBD,
        rdMolDescriptors.CalcNumHBA,
        rdMolDescriptors.CalcNumRotatableBonds,
        rdMolDescriptors.CalcNumAromaticRings,
        rdMolDescriptors.CalcNumAliphaticRings,
        Descriptors.HeavyAtomCount,
        rdMolDescriptors.CalcNumSaturatedRings,
        Descriptors.FractionCSP3,
        rdMolDescriptors.CalcLabuteASA,
        GraphDescriptors.BalabanJ,
        GraphDescriptors.BertzCT,
        GraphDescriptors.Chi0n,
        GraphDescriptors.Chi1n,
        rdMolDescriptors.CalcKappa1,
        rdMolDescriptors.CalcKappa2,
        rdMolDescriptors.CalcNumRings,
        Descriptors.NumRadicalElectrons,
        Descriptors.MaxPartialCharge,
    ]
    values = [_safe_descriptor(fn, mol) for fn in fns]
    return np.array(values, dtype=np.float32)


def compute_morgan_fp(
    mol: Chem.rdchem.Mol,
    radius: int = MORGAN_RADIUS,
    n_bits: int = MORGAN_NBITS,
) -> np.ndarray:
    """
    Compute a binary Morgan (ECFP) fingerprint.

    This is intentionally kept separate from the scalar physchem vector:
    the GNN path does not need it, but the MLP ablation path and virtual
    screening ranking benefit significantly from it.

    Returns
    -------
    np.ndarray of shape (n_bits,) dtype=float32  (binary, 0/1)
    """
    try:
        from rdkit.Chem import rdMolDescriptors as _rmd
        fp = _rmd.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return np.zeros(n_bits, dtype=np.float32)



def generate_3d_conformer(
    mol: Chem.rdchem.Mol,
    seed: int = 42,
    max_attempts: int = 10,
    num_confs: int = 1,
) -> Optional[np.ndarray]:
    """
    Embed a molecule in 3D using ETKDG and minimise with UFF.

    Parameters
    ----------
    mol          : RDKit molecule (no explicit Hs required; they are added
                   temporarily for embedding)
    seed         : random seed for ETKDG
    max_attempts : number of embedding attempts before giving up
    num_confs    : number of conformers to generate (lowest energy is kept)

    Returns
    -------
    np.ndarray of shape (N_heavy_atoms, 3), centred at centroid.
    None if embedding or minimisation fails.

    Notes
    -----
    Explicit Hs are added before embedding and stripped from the returned
    coordinate matrix so it aligns with the heavy-atom graph.
    """
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 0   # use all available CPUs

    conf_ids = AllChem.EmbedMultipleConfs(
        mol_h,
        numConfs=num_confs,
        params=params,
    )
    if len(conf_ids) == 0:
        # Retry with random coordinates as seeds
        params.useRandomCoords = True
        for attempt in range(max_attempts):
            params.randomSeed = seed + attempt
            conf_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=1, params=params)
            if conf_ids:
                break
    if len(conf_ids) == 0:
        log.debug("3D embedding failed for molecule with %d atoms.", mol.GetNumAtoms())
        return None

    # Minimise with UFF
    energies = []
    for cid in conf_ids:
        ff = AllChem.UFFGetMoleculeForceField(mol_h, confId=cid)
        if ff is None:
            energies.append((float("inf"), cid))
            continue
        ff.Minimize(maxIts=1000)
        energies.append((ff.CalcEnergy(), cid))

    best_cid = min(energies, key=lambda t: t[0])[1]

    # Extract heavy-atom coordinates
    conf = mol_h.GetConformer(best_cid)
    heavy_atom_indices = [
        atom.GetIdx()
        for atom in mol_h.GetAtoms()
        if atom.GetAtomicNum() != 1
    ]
    coords = np.array(
        [conf.GetAtomPosition(idx) for idx in heavy_atom_indices],
        dtype=np.float32,
    )   # shape: (N_heavy, 3)

    # Centre at centroid
    coords -= coords.mean(axis=0)
    return coords


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Main featuriser class
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class LigandFeaturizer:
    """
    Convert a SMILES string to a full ligand feature bundle.

    Attributes
    ----------
    compute_3d : bool
        Whether to attempt 3D conformer generation (expensive; default True).
    conformer_seed : int
        ETKDG random seed.

    Returns (via ``featurize``)
    ---------------------------
    torch_geometric.data.Data with extra attributes:
        .physchem   : float tensor (PHYSCHEM_DIM,)
        .coords     : float tensor (N_atoms, 3) or None
        .inchikey   : str
        .smiles     : str (canonical)
    """

    def __init__(self, compute_3d: bool = True, conformer_seed: int = 42):
        self.compute_3d     = compute_3d
        self.conformer_seed = conformer_seed

    # ------------------------------------------------------------------
    def featurize(
        self,
        smiles: str,
        inchikey: str = "",
    ) -> Optional[Data]:
        """
        Featurise a single SMILES string.

        Parameters
        ----------
        smiles   : canonical SMILES (should already be standardised by module1)
        inchikey : molecule identifier (stored on Data object)

        Returns
        -------
        Data object or None if the SMILES cannot be parsed.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            log.warning("Cannot parse SMILES: %s", smiles)
            return None

        # Remove explicit Hs for the graph (GNNs typically use heavy-atom graphs)
        mol_no_h = Chem.RemoveHs(mol)
        if mol_no_h.GetNumAtoms() == 0:
            log.warning("Molecule has 0 heavy atoms: %s", smiles)
            return None

        # 1. Graph features
        data = mol_to_graph(mol_no_h, inchikey=inchikey)
        data.smiles = smiles

        # 2. Physicochemical features
        physchem = compute_physchem(mol_no_h)
        data.physchem = torch.tensor(physchem, dtype=torch.float)

        # NaN safety: warn and impute zeros so no NaNs silently pass through.
        # Caller should still normalize before training.
        if torch.isnan(data.physchem).any():
            n_physchem_nans = int(torch.isnan(data.physchem).sum().item())
            log.warning(
                "Molecule %s has %d NaN physicochemical values; imputing 0.0",
                inchikey or smiles,
                n_physchem_nans,
            )
            data.physchem = torch.nan_to_num(data.physchem, nan=0.0)

        # 2b. Morgan fingerprint (ECFP4, 1024-bit) â€” stored separately.
        #     Used by the MLP ablation encoder and virtual screening ranking.
        data.morgan_fp = torch.tensor(
            compute_morgan_fp(mol_no_h), dtype=torch.float
        )

        # 3. Optional 3D conformer
        if self.compute_3d:
            coords = generate_3d_conformer(mol_no_h, seed=self.conformer_seed)
            if coords is not None:
                data.coords = torch.tensor(coords, dtype=torch.float)
            else:
                data.coords = None
        else:
            data.coords = None

        return data

    # ------------------------------------------------------------------
    def featurize_batch(
        self,
        smiles_list: list[str],
        inchikey_list: Optional[list[str]] = None,
    ) -> dict[str, Optional[Data]]:
        """
        Featurise a list of SMILES strings.

        Returns
        -------
        dict mapping each SMILES to its Data object (or None on failure).
        """
        if inchikey_list is None:
            inchikey_list = [""] * len(smiles_list)

        results: dict[str, Optional[Data]] = {}
        for smi, ik in zip(smiles_list, inchikey_list):
            results[smi] = self.featurize(smi, inchikey=ik)

        n_ok   = sum(1 for v in results.values() if v is not None)
        n_fail = len(results) - n_ok
        log.info("Featurized %d molecules: %d OK, %d failed.", len(results), n_ok, n_fail)
        return results


class LigandFeatureStore(Mapping[str, Data]):
    """Disk-backed ligand feature store. Does not keep all Data objects in memory."""

    def __init__(
        self,
        inchikeys: Iterable[str],
        output_dir: Path,
        cache_size: int = 128,
        load_retries: int = 3,
        retry_delay: float = 0.05,
    ):
        self._keys = list(inchikeys)
        self.output_dir = Path(output_dir)
        self.cache_size = max(0, int(cache_size))
        self.load_retries = max(1, int(load_retries))
        self.retry_delay = max(0.0, float(retry_delay))
        self._cache: OrderedDict[str, Data] = OrderedDict()

    def _safe_load(self, path: Path, key: str) -> Data:
        last_exc: Exception | None = None
        for attempt in range(1, self.load_retries + 1):
            try:
                return torch.load(path, weights_only=False)
            except OSError as exc:
                last_exc = exc
                if exc.errno != errno.EINVAL or attempt >= self.load_retries:
                    break
                time.sleep(self.retry_delay * attempt)
            except Exception as exc:
                last_exc = exc
                break

        raise OSError(
            f"Failed to load ligand features for inchikey='{key}' from '{path}'"
        ) from last_exc

    def __getitem__(self, key: str) -> Data:
        path = self.output_dir / f"{key}.pt"
        if not path.exists():
            raise KeyError(key)
        if self.cache_size > 0 and key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        data = self._safe_load(path, key)

        if self.cache_size > 0:
            self._cache[key] = data
            self._cache.move_to_end(key)
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        return data

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def keys(self):
        return self._keys


def load_ligand_feature_store(path: str | Path):
    """
    Load either a legacy single-file torch store or the disk-backed directory store.
    """
    path = Path(path)
    if path.is_dir():
        return LigandFeatureStore([p.stem for p in path.glob("*.pt")], path)

    output_dir = path.with_suffix("") if path.suffix == ".pt" else path
    output_dir = Path(output_dir)
    if output_dir.is_dir():
        return LigandFeatureStore([p.stem for p in output_dir.glob("*.pt")], output_dir)

    return torch.load(path, weights_only=False)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Internal helper for parallel featurization
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _featurize_and_save(args):
    """Process a single row in a worker process and save to disk."""
    smiles, inchikey, output_dir, compute_3d, conformer_seed = args
    featurizer = LigandFeaturizer(compute_3d=compute_3d, conformer_seed=conformer_seed)
    try:
        data = featurizer.featurize(smiles, inchikey=inchikey)
        if data is None:
            return inchikey, False, "parse_failed"
        path = Path(output_dir) / f"{inchikey}.pt"
        torch.save(data, path)
        return inchikey, True, "ok"
    except Exception as exc:
        return inchikey, False, str(exc)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Feature store builder (processes the full clean dataset)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_ligand_feature_store(
    dataset_path: str = "./pipeline_outputs/dataset_clean.parquet",
    output_path: str = "./pipeline_outputs/ligand_features",
    compute_3d: bool = True,
    conformer_seed: int = 42,
    use_cache: bool = True,
) -> dict[str, Data]:
    """
    Build and persist a {inchikey: Data} feature store from the clean dataset.

    Parameters
    ----------
    dataset_path  : path to dataset_clean.parquet
    output_path   : path to save the feature store (torch .pt file)
    compute_3d    : enable 3D conformer generation
    conformer_seed: ETKDG seed
    use_cache     : if True and output_path exists, reload from disk

    Returns
    -------
    dict mapping InChIKey â†’ torch_geometric.data.Data
    """
    import torch
    from pathlib import Path
    from concurrent.futures import ProcessPoolExecutor, as_completed

    output_path = Path(output_path)
    output_dir = output_path.with_suffix("") if output_path.suffix == ".pt" else output_path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Legacy loader path: allow old .pt dictionary file to be used if present.
    if use_cache and output_path.is_file():
        log.info("Loading old-style feature store from cache: %s", output_path)
        legacy_store = torch.load(output_path, weights_only=False)
        for ik, data in legacy_store.items():
            torch.save(data, output_dir / f"{ik}.pt")
        return LigandFeatureStore(legacy_store.keys(), output_dir)

    log.info("Loading clean dataset from %s â€¦", dataset_path)
    df = __import__("pandas").read_parquet(dataset_path)

    # Deduplicate: one feature object per unique InChIKey
    unique_mols = df[["inchikey", "smiles"]].drop_duplicates(subset="inchikey")
    log.info("Unique molecules to featurize: %d", len(unique_mols))

    existing_keys = set()
    if output_dir.exists():
        existing_keys = {p.stem for p in output_dir.glob("*.pt")}
        if use_cache and existing_keys and len(existing_keys) >= len(unique_mols):
            log.info("Loading disk-backed ligand feature store from cache: %s", output_dir)
            return LigandFeatureStore(existing_keys, output_dir)
        if existing_keys:
            log.info(
                "Resuming ligand feature store in %s (%d already present, %d remaining).",
                output_dir,
                len(existing_keys),
                max(len(unique_mols) - len(existing_keys), 0),
            )

    task_args = [
        (row.smiles, row.inchikey, output_dir, compute_3d, conformer_seed)
        for row in unique_mols.itertuples(index=False)
        if row.inchikey not in existing_keys
    ]

    n_total = len(task_args)
    n_ok = len(existing_keys)
    n_fail = 0
    errors = []

    if n_total == 0:
        log.info("No ligand featurization work remaining.")
        return LigandFeatureStore([p.stem for p in output_dir.glob("*.pt")], output_dir)

    try:
        with ProcessPoolExecutor() as executor:
            future_to_ik = {
                executor.submit(_featurize_and_save, arg): arg[1]
                for arg in task_args
            }
            for i, future in enumerate(as_completed(future_to_ik), 1):
                ik = future_to_ik[future]
                try:
                    _, success, message = future.result()
                    if success:
                        n_ok += 1
                    else:
                        n_fail += 1
                        errors.append((ik, message))
                except Exception as exc:
                    n_fail += 1
                    errors.append((ik, str(exc)))

                if i % 500 == 0 or i == n_total:
                    log.info(
                        "  Progress: %d / %d (ok: %d, fail: %d)",
                        i, n_total, n_ok, n_fail,
                    )
    except (PermissionError, OSError) as exc:
        log.warning(
            "Process-based featurization unavailable in this environment (%s). "
            "Falling back to single-process execution.",
            exc,
        )
        for i, arg in enumerate(task_args, 1):
            ik = arg[1]
            try:
                _, success, message = _featurize_and_save(arg)
                if success:
                    n_ok += 1
                else:
                    n_fail += 1
                    errors.append((ik, message))
            except Exception as inner_exc:
                n_fail += 1
                errors.append((ik, str(inner_exc)))

            if i % 500 == 0 or i == n_total:
                log.info(
                    "  Progress: %d / %d (ok: %d, fail: %d)",
                    i, n_total, n_ok, n_fail,
                )

    if errors:
        log.warning("%d molecules failed during parallel featurization (sample): %s",
                    len(errors), errors[:5])

    log.info("Feature store built: %d molecules, %d failures", n_ok, n_fail)
    log.info("Saved individual feature files under %s", output_dir)

    return LigandFeatureStore([p.stem for p in output_dir.glob("*.pt")], output_dir)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Convenience: physicochemical imputation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def impute_physchem_store(
    store,
    strategy: str = "median",
):
    """
    In-place impute NaN physicochemical features across all molecules.

    Works with both plain dict stores and disk-backed LigandFeatureStore.

    Parameters
    ----------
    store    : feature store produced by build_ligand_feature_store
    strategy : "median" (default) or "zero"

    Returns the same store (mutated in-place where possible).
    """
    import torch

    # Materialise all physchem vectors; skip entries with wrong dim
    pc_list: list[torch.Tensor] = []
    key_list: list[str] = []
    for ik in store:
        try:
            d = store[ik]
            if hasattr(d, "physchem") and d.physchem is not None:
                if d.physchem.ndim == 1 and d.physchem.shape[0] == PHYSCHEM_DIM:
                    pc_list.append(d.physchem)
                    key_list.append(ik)
        except Exception:
            continue

    if not pc_list:
        log.warning("impute_physchem_store: no valid physchem tensors found â€” skipping.")
        return store

    all_pc = torch.stack(pc_list)   # (N, PHYSCHEM_DIM)

    if strategy == "median":
        arr = all_pc.numpy()
        col_medians = np.nanmedian(arr, axis=0)          # (PHYSCHEM_DIM,)
        impute_vals = torch.tensor(col_medians, dtype=torch.float)
    else:
        impute_vals = torch.zeros(PHYSCHEM_DIM, dtype=torch.float)

    # Apply imputation
    n_imputed = 0
    for ik in key_list:
        try:
            data = store[ik]
            nan_mask = torch.isnan(data.physchem)
            if nan_mask.any():
                data.physchem[nan_mask] = impute_vals[nan_mask]
                n_imputed += nan_mask.sum().item()
                # For disk-backed stores we need to re-save
                if isinstance(store, LigandFeatureStore):
                    torch.save(data, store.output_dir / f"{ik}.pt")
        except Exception:
            continue

    n_remaining = sum(
        torch.isnan(store[ik].physchem).sum().item()
        for ik in key_list
        if hasattr(store[ik], "physchem")
    )
    log.info(
        "Physicochemical imputation: %d NaNs replaced. Remaining NaNs: %d",
        n_imputed, n_remaining,
    )
    return store


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Entry point
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build ligand feature store.")
    parser.add_argument("--dataset",  default="./pipeline_outputs/dataset_clean.parquet", help="Clean parquet")
    parser.add_argument("--output",   default="./pipeline_outputs/ligand_features",        help="Output store path")
    parser.add_argument("--no-3d",    action="store_true",             help="Skip 3D conformers")
    parser.add_argument("--no-cache", action="store_true",             help="Force rebuild")
    args = parser.parse_args()

    store = build_ligand_feature_store(
        dataset_path=args.dataset,
        output_path=args.output,
        compute_3d=not args.no_3d,
        use_cache=not args.no_cache,
    )

    # Quick sanity check
    sample_ik = next(iter(store))
    sample = store[sample_ik]
    print(f"\nSample InChIKey : {sample.inchikey}")
    print(f"Atom features   : {sample.x.shape}   (expected: N_atoms Ã— {ATOM_FEAT_DIM})")
    print(f"Edge index      : {sample.edge_index.shape}")
    print(f"Edge attr       : {sample.edge_attr.shape}   (expected: N_edges Ã— {BOND_FEAT_DIM})")
    print(f"Physchem        : {sample.physchem.shape}   (expected: {PHYSCHEM_DIM})")
    sample_coords = getattr(sample, "coords", None)
    print(f"3D coords       : {sample_coords.shape if sample_coords is not None else 'None'}")
    print(f"\nAtom feat dim   : {ATOM_FEAT_DIM}")
    print(f"Bond feat dim   : {BOND_FEAT_DIM}")
    print(f"Physchem dim    : {PHYSCHEM_DIM}")

