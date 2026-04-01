"""
module1_dataset_builder.py
==========================
TITLE
Kinase-ligand dataset builder and registry cleaning stage.

PURPOSE
This module turns the raw kinase registry plus external assay sources into a
clean, sampled, auditable dataset that the rest of the pipeline can trust.

WHAT IT DOES
- Reads and validates the kinase registry.
- Filters targets by protein-feasibility status.
- Fetches IC50 observations from supported sources.
- Standardizes ligands and deduplicates cross-source records.
- Aggregates pIC50 statistics and applies exact per-kinase sampling rules.
- Exports dataset, retained-target, provenance, and audit artefacts.

HOW IT WORKS
1. Load the authoritative registry columns.
2. Run protein feasibility gating through module3.
3. Fetch assay records only for retained targets.
4. Standardize ligands and resolve duplicate-source conflicts.
5. Sample each kinase according to the requested active:inactive ratio.
6. Write final dataset and supporting audit tables.

INPUT CONTRACT
- Excel workbook with `target`, `uniprot_id`, and `pdb_id`.
- Network/API availability if cache is unavailable.

OUTPUT CONTRACT
- `dataset_clean.parquet`
- `retained_targets.csv`
- `registry_clean.parquet`
- dataset summary, duplicate audit, provenance, sampling, and fetch logs

DEPENDENCIES
- pandas, numpy, RDKit, requests
- module3_protein_features.py

CRITICAL ASSUMPTIONS
- Registry metadata is authoritative.
- Per-kinase quota sampling is the intended scientific design.
- Source priority is strict: ChEMBL > BindingDB > PubChem.

FAILURE MODES
- Missing registry columns
- Zero retained targets after feasibility filtering
- Empty sampled dataset
- External fetch failures without cache

SAFETY CHECKS IMPLEMENTED
- Registry schema checks
- Explicit target drop reasons
- Duplicate/provenance audits
- Cache-aware rebuild logic

HOW TO RUN
- `python module1_dataset_builder.py --excel ML.xlsx --out-dir ./pipeline_outputs --ratio_mode 1:2`

HOW IT CONNECTS TO PIPELINE
It produces the cleaned dataset and retained target list consumed by feature
generation, training, evaluation, and the Streamlit inference app.
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from module3_protein_features import run_feasibility_filter_stage
from progress_utils import progress_iter

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Logging
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("module1")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Constants
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CHEMBL_BASE    = "https://www.ebi.ac.uk/chembl/api/data"
BINDINGDB_BASE = "https://www.bindingdb.org/axis2/services/BDBService"
PUBCHEM_BASE   = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# IC50 ONLY â€” strict measurement type filter
IC50_ONLY: set[str] = {"IC50"}

# Unit â†’ nM conversion table
UNIT_TO_NM: dict[str, float] = {
    "nM": 1.0,     "nm": 1.0,
    "uM": 1_000.0, "um": 1_000.0, "ÂµM": 1_000.0, "Âµm": 1_000.0,
    "mM": 1_000_000.0,
    "pM": 0.001,   "pm": 0.001,
    "M":  1_000_000_000.0,
}

# pIC50 physical validity range
PIC50_MIN = 2.0    # 10 mM â€” pharmacologically dead
PIC50_MAX = 16.0   # 10 fM â€” instrument artefact

# Source priority (lower index = higher priority)
SOURCE_PRIORITY: dict[str, int] = {
    "chembl":    0,
    "bindingdb": 1,
    "pubchem":   2,
}

# Exact quotas (hard constraints)
EXACT_ACTIVES = 15
RATIO_INACTIVES: dict[str, int] = {"1:1": 15, "1:2": 30, "1:3": 45}

# Activity label: pIC50 â‰¥ threshold â†’ active
# Using a globally consistent cut-off (6.0 = 1 ÂµM) per CHEMBL convention.
# This can be overridden via --active_pic50_cutoff.
DEFAULT_ACTIVE_CUTOFF_PIC50 = 6.0

PAGE_SIZE        = 1000
MAX_WORKERS      = 8
REQUEST_RETRIES  = 5
REQUEST_BACKOFF  = 2.0   # seconds; doubles on retry
MAX_PUBCHEM_AIDS = 15    # cap assay IDs to avoid runaway PubChem calls

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SMILES standardisation (module-level singletons â€” thread-safe for reading)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER        = rdMolStandardize.Uncharger()
_NORMALIZER       = rdMolStandardize.Normalizer()


def standardise_smiles(smi: str) -> Optional[str]:
    """
    Canonicalise, desalt, and normalise a SMILES string.
    Returns canonical SMILES (stereochemistry preserved) or None on failure.
    """
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi.strip())
    if mol is None:
        return None
    try:
        mol = _NORMALIZER.normalize(mol)
        mol = _LARGEST_FRAGMENT.choose(mol)
        mol = _UNCHARGER.uncharge(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def smiles_to_inchikey(smi: str) -> Optional[str]:
    """Return InChIKey for a standardised SMILES, or None on failure."""
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        inchi = Chem.MolToInchi(mol)
        return Chem.InchiToInchiKey(inchi) if inchi else None
    except Exception:
        return None


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Unit conversion
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def convert_to_nm(value: float, unit: str) -> Optional[float]:
    """Convert an IC50 measurement to nM. Returns None for unknown units."""
    unit = unit.strip()
    multiplier = UNIT_TO_NM.get(unit)
    if multiplier is None:
        for k, v in UNIT_TO_NM.items():
            if k.lower() == unit.lower():
                multiplier = v
                break
    if multiplier is None:
        return None
    val = value * multiplier
    return val if val > 0 else None


def nm_to_pic50(nm: float) -> float:
    """pIC50 = 9 âˆ’ log10(IC50_nM)."""
    return 9.0 - np.log10(nm)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# HTTP helper with exponential backoff
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _get_json(
    url: str,
    params: Optional[dict] = None,
    timeout: int = 60,
) -> Optional[dict]:
    delay = REQUEST_BACKOFF
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == REQUEST_RETRIES:
                log.warning("HTTP GET failed [%s]: %s", url[:80], exc)
                return None
            time.sleep(min(delay * 2, 60.0))
            delay *= 2
    return None


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  STAGE 1 â€” Registry loading + protein-feasibility filtering             â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_registry(excel_path: str | Path) -> pd.DataFrame:
    """
    Load the target registry from ML.xlsx.

    ONLY the columns [target, uniprot_id, pdb_id] are treated as authoritative.
    All other columns are silently ignored.

    Returns a DataFrame with exactly those three columns, deduplicated by
    uniprot_id.
    """
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")

    raw = pd.read_excel(path, dtype=str)

    # Normalise column names
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]

    required = {"target", "uniprot_id", "pdb_id"}
    missing  = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"ML.xlsx is missing required columns: {missing}. "
            f"Found: {list(raw.columns)}"
        )

    df = raw[["target", "uniprot_id", "pdb_id"]].copy()
    df = df.dropna(subset=["uniprot_id"])
    df["uniprot_id"] = df["uniprot_id"].str.strip()
    df["target"]     = df["target"].str.strip()
    df["pdb_id"]     = df["pdb_id"].str.strip()

    before = len(df)
    df = df.drop_duplicates(subset=["uniprot_id"])
    after  = len(df)

    log.info(
        "Registry: loaded %d rows, %d unique UniProt IDs (dropped %d duplicates)",
        before, after, before - after,
    )
    return df.reset_index(drop=True)


def run_protein_feasibility_filter(
    registry: pd.DataFrame,
    out_dir: Path,
    workers: int = MAX_WORKERS,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Delegate Stage 1 feasibility filtering to the shared module3 implementation.
    """
    return run_feasibility_filter_stage(
        registry=registry,
        out_dir=out_dir,
        workers=workers,
        use_cache=use_cache,
    )


# â”€â”€ Sub-helpers called by the feasibility filter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _fetch_uniprot_sequence(uid: str) -> Optional[str]:
    """Retrieve the canonical protein sequence from UniProt REST API."""
    url  = f"https://rest.uniprot.org/uniprotkb/{uid}.fasta"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        lines = resp.text.strip().split("\n")
        seq   = "".join(l for l in lines if not l.startswith(">"))
        return seq if seq else None
    except Exception:
        return None


def _check_alphafold(uid: str) -> bool:
    """Return True if AlphaFold has a structure for this UniProt ID."""
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{uid}"
    try:
        resp = requests.get(url, timeout=20)
        return resp.status_code == 200 and bool(resp.json())
    except Exception:
        return False


def _check_klifs_pocket(uid: str) -> int:
    """
    Return the number of KLIFS pocket residues available for this UniProt ID.
    Returns 0 if KLIFS has no mapping (target should be dropped).

    Uses the KLIFS REST API directly (no opencadd client required here).
    """
    url    = "https://klifs.net/api/kinase_information"
    params = {"kinase_ID": "", "species": "Human"}
    # KLIFS uniprot search endpoint
    url2    = f"https://klifs.net/api/kinases/uniprot?uniprot={uid}"
    try:
        resp = requests.get(url2, timeout=20)
        if resp.status_code != 200:
            return 0
        data = resp.json()
        if not data:
            return 0
        # If we found a KLIFS entry, assume 85 pocket residues (canonical)
        return 85
    except Exception:
        return 0


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  STAGE 2 â€” Ligand fetching (retained targets only)                      â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# â”€â”€ SOURCE 1: ChEMBL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fetch_chembl_ic50(uniprot_id: str, quota: int) -> pd.DataFrame:
    """
    Fetch IC50 records from ChEMBL for one UniProt accession.

    quota: stop early once this many valid records are collected (prevents
           unnecessary over-fetching). Use a generous multiple of the final
           quota to allow for downstream filtering losses.
    """
    target_data = _get_json(
        f"{CHEMBL_BASE}/target.json",
        {
            "target_components__accession": uniprot_id,
            "target_type": "SINGLE PROTEIN",
            "limit": 20,
            "format": "json",
        },
    )
    if not target_data:
        return pd.DataFrame()

    chembl_tids = [t["target_chembl_id"] for t in target_data.get("targets", [])]
    if not chembl_tids:
        return pd.DataFrame()

    rows: list[dict] = []
    for tid in chembl_tids:
        if len(rows) >= quota:
            break
        offset = 0
        while True:
            data = _get_json(
                f"{CHEMBL_BASE}/activity.json",
                {
                    "target_chembl_id":  tid,
                    "standard_type":     "IC50",
                    "standard_relation": "=",
                    "limit":             PAGE_SIZE,
                    "offset":            offset,
                    "format":            "json",
                },
            )
            if not data:
                break

            activities = data.get("activities", [])
            for rec in activities:
                smi     = rec.get("canonical_smiles") or rec.get("molecule_smiles")
                std_val = rec.get("standard_value")
                pchembl = rec.get("pchembl_value")

                if not smi or std_val is None:
                    continue

                try:
                    if pchembl is not None:
                        pic50   = float(pchembl)
                        ic50_nm = 10 ** (9 - pic50)
                    else:
                        nm = convert_to_nm(float(std_val), rec.get("standard_units", ""))
                        if nm is None:
                            continue
                        ic50_nm = nm
                        pic50   = nm_to_pic50(nm)
                except (TypeError, ValueError):
                    continue

                if not (PIC50_MIN <= pic50 <= PIC50_MAX):
                    continue

                rows.append({
                    "smiles_raw":  smi,
                    "uniprot_id":  uniprot_id,
                    "ic50_nm":     ic50_nm,
                    "pIC50":       pic50,
                    "source":      "chembl",
                    "source_priority": SOURCE_PRIORITY["chembl"],
                    "assay_id":    rec.get("assay_chembl_id", ""),
                    "confidence":  rec.get("confidence_score", np.nan),
                })

            total  = data.get("page_meta", {}).get("total_count", len(activities))
            offset += PAGE_SIZE
            if offset >= total or len(rows) >= quota:
                break

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# â”€â”€ SOURCE 2: BindingDB â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fetch_bindingdb_ic50(uniprot_id: str) -> pd.DataFrame:
    """Fetch IC50 records from BindingDB for one UniProt accession."""
    data = _get_json(
        f"{BINDINGDB_BASE}/getLigandsByUniprot",
        {"uniprot": uniprot_id, "response": "json"},
        timeout=90,
    )
    if not data:
        return pd.DataFrame()

    affinities = (
        data
        .get("getLigandsByUniprotResponse", {})
        .get("affinities", [])
    )
    if not affinities:
        return pd.DataFrame()

    rows: list[dict] = []
    for rec in affinities:
        if "IC50" not in str(rec.get("affinity_type", "")).upper():
            continue

        smi     = rec.get("ligand_smiles", "")
        val_str = str(rec.get("affinity", "")).strip()
        unit    = rec.get("affinity_unit", "nM")

        if not smi or not val_str:
            continue
        if val_str.startswith(">") or val_str.startswith("<"):
            continue

        try:
            val = float(val_str.replace(",", ""))
        except ValueError:
            continue

        nm = convert_to_nm(val, unit)
        if nm is None:
            continue
        pic50 = nm_to_pic50(nm)
        if not (PIC50_MIN <= pic50 <= PIC50_MAX):
            continue

        rows.append({
            "smiles_raw":      smi,
            "uniprot_id":      uniprot_id,
            "ic50_nm":         nm,
            "pIC50":           pic50,
            "source":          "bindingdb",
            "source_priority": SOURCE_PRIORITY["bindingdb"],
            "assay_id":        rec.get("bindingdb_monomerid", ""),
            "confidence":      np.nan,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# â”€â”€ SOURCE 3: PubChem â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fetch_pubchem_ic50(uniprot_id: str) -> pd.DataFrame:
    """Fetch IC50 records from PubChem BioAssay for one UniProt accession."""
    search = _get_json(
        f"{PUBCHEM_BASE}/assay/target/ProteinAccession/{uniprot_id}/aids/JSON",
    )
    if not search:
        return pd.DataFrame()

    aids = search.get("IdentifierList", {}).get("AID", [])[:MAX_PUBCHEM_AIDS]
    if not aids:
        return pd.DataFrame()

    rows: list[dict] = []
    for aid in aids:
        assay_data = _get_json(f"{PUBCHEM_BASE}/assay/aid/{aid}/concise/JSON")
        if not assay_data:
            continue

        table    = assay_data.get("Table", {})
        columns  = table.get("Column", [])
        row_data = table.get("Row", [])
        if not columns or not row_data:
            continue

        col_map     = {c.lower(): i for i, c in enumerate(columns)}
        cid_idx     = col_map.get("cid")
        outcome_idx = col_map.get("activity outcome")
        ic50_idx    = next((col_map[k] for k in col_map if "ic50" in k), None)

        if any(x is None for x in [cid_idx, outcome_idx, ic50_idx]):
            continue

        active_cids: list[int] = []
        ic50_vals: dict[int, float] = {}
        for r in row_data:
            cells = r.get("Cell", [])
            if len(cells) <= max(cid_idx, outcome_idx, ic50_idx):
                continue
            if str(cells[outcome_idx]).lower() != "active":
                continue
            try:
                cid  = int(cells[cid_idx])
                val  = float(str(cells[ic50_idx]).replace(",", ""))
                active_cids.append(cid)
                ic50_vals[cid] = val
            except (ValueError, TypeError):
                continue

        if not active_cids:
            continue

        for batch_start in range(0, len(active_cids), 100):
            batch   = active_cids[batch_start: batch_start + 100]
            cid_str = ",".join(str(c) for c in batch)
            smi_data = _get_json(
                f"{PUBCHEM_BASE}/compound/cid/{cid_str}/property/IsomericSMILES/JSON"
            )
            if not smi_data:
                continue

            for p in smi_data.get("PropertyTable", {}).get("Properties", []):
                cid = p.get("CID")
                smi = p.get("IsomericSMILES", "")
                if not cid or not smi:
                    continue
                ic50_uM = ic50_vals.get(cid)
                if ic50_uM is None:
                    continue
                nm    = ic50_uM * 1_000.0
                pic50 = nm_to_pic50(nm)
                if not (PIC50_MIN <= pic50 <= PIC50_MAX):
                    continue
                rows.append({
                    "smiles_raw":      smi,
                    "uniprot_id":      uniprot_id,
                    "ic50_nm":         nm,
                    "pIC50":           pic50,
                    "source":          "pubchem",
                    "source_priority": SOURCE_PRIORITY["pubchem"],
                    "assay_id":        str(aid),
                    "confidence":      np.nan,
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  STAGE 3 â€” Standardisation + source-priority deduplication              â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def standardise_and_deduplicate(
    raw: pd.DataFrame,
    active_pic50_cutoff: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    1. Standardise SMILES (canonicalise, desalt, normalise).
    2. Generate InChIKey (molecular identity).
    3. Drop invalid / failed molecules EARLY.
    4. Apply source priority deduplication:
          For each (uniprot_id, inchikey) pair keep only the highest-priority
          source record. Discard all lower-priority duplicates.
    5. Aggregate per (uniprot_id, inchikey): pIC50_median, pIC50_std, n_meas.
    6. Assign activity_label using the global pIC50 cutoff.

    Returns (clean_df, duplicate_audit_df).
    """
    log.info("Standardising %d raw records ...", len(raw))

    # â”€â”€ Standardise SMILES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    raw["canonical_smiles"] = raw["smiles_raw"].apply(standardise_smiles)
    n_invalid = raw["canonical_smiles"].isna().sum()
    log.info("  Invalid / unparseable SMILES dropped: %d", n_invalid)
    raw = raw.dropna(subset=["canonical_smiles"]).copy()

    # â”€â”€ InChIKey â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    raw["inchikey"] = raw["canonical_smiles"].apply(smiles_to_inchikey)
    n_no_key = raw["inchikey"].isna().sum()
    log.info("  Records with no InChIKey dropped: %d", n_no_key)
    raw = raw.dropna(subset=["inchikey"]).copy()

    # â”€â”€ Source-priority deduplication â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # For each (uniprot_id, inchikey, source) group: keep one record (median).
    agg_per_source = (
        raw.groupby(["uniprot_id", "inchikey", "source", "source_priority"])
        .agg(
            pIC50_median=("pIC50",   "median"),
            ic50_nm_median=("ic50_nm", "median"),
            n_measurements=("pIC50",  "count"),
            pIC50_std=("pIC50",       "std"),
            assay_id=("assay_id",     "first"),
            canonical_smiles=("canonical_smiles", "first"),
        )
        .reset_index()
    )

    # Build duplicate audit BEFORE priority filtering
    dup_counts = (
        agg_per_source.groupby(["uniprot_id", "inchikey"])["source"]
        .apply(list)
        .reset_index()
        .rename(columns={"source": "all_sources"})
    )
    dup_counts["n_sources"] = dup_counts["all_sources"].apply(len)
    dup_counts["is_cross_source_duplicate"] = dup_counts["n_sources"] > 1

    dup_audit = dup_counts.copy()
    dup_audit["all_sources"] = dup_audit["all_sources"].apply(
        lambda x: ",".join(sorted(x))
    )

    # Keep highest-priority source per (uniprot_id, inchikey)
    best = (
        agg_per_source
        .sort_values("source_priority")
        .groupby(["uniprot_id", "inchikey"])
        .first()
        .reset_index()
    )
    best.rename(
        columns={
            "pIC50_median":    "pIC50",
            "ic50_nm_median":  "ic50_nm",
            "source":          "assay_source",
        },
        inplace=True,
    )

    n_cross_dup = dup_counts["is_cross_source_duplicate"].sum()
    log.info(
        "  Cross-source duplicates resolved: %d InChIKey pairs "
        "(kept highest-priority source)",
        n_cross_dup,
    )

    # â”€â”€ Impute missing pIC50_std (singleton records) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    global_std = best["pIC50_std"].median()
    n_nan_std  = best["pIC50_std"].isna().sum()
    best["pIC50_std"] = best["pIC50_std"].fillna(global_std)
    log.info("  Singleton pIC50_std imputed (global median %.3f): %d records",
             global_std, n_nan_std)

    # â”€â”€ Activity label â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    best["activity_label"] = np.where(
        best["pIC50"] >= active_pic50_cutoff, "active", "inactive"
    )

    return best, dup_audit


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  STAGE 4 â€” Exact per-kinase sampling with ratio_mode control            â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def sample_kinase_dataset(
    df: pd.DataFrame,
    ratio_mode: str,
    seed: int,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Sample EXACTLY:
        actives   = EXACT_ACTIVES     (15)
        inactives = RATIO_INACTIVES[ratio_mode]  (15 / 30 / 45)

    Per kinase (uniprot_id).

    Rules:
        - No replacement.
        - Deterministic (fixed seed).
        - If a kinase cannot fill BOTH quotas â†’ drop + log.
        - Sampling is stratified within each label class.

    Returns (sampled_df, sampling_log).
    """
    n_inactives = RATIO_INACTIVES[ratio_mode]
    rng = np.random.default_rng(seed)

    sampled_parts: list[pd.DataFrame] = []
    sample_log: list[dict] = []

    for uid, group in df.groupby("uniprot_id"):
        actives_pool   = group[group["activity_label"] == "active"]
        inactives_pool = group[group["activity_label"] == "inactive"]

        n_avail_active   = len(actives_pool)
        n_avail_inactive = len(inactives_pool)

        entry = {
            "uniprot_id":       uid,
            "n_avail_actives":  n_avail_active,
            "n_avail_inactives":n_avail_inactive,
            "n_sampled_actives":  0,
            "n_sampled_inactives":0,
            "status":           "retained",
            "drop_reason":      "",
        }

        # â”€â”€ Quota check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if n_avail_active < EXACT_ACTIVES:
            entry["status"]      = "dropped"
            entry["drop_reason"] = (
                f"insufficient_actives: need {EXACT_ACTIVES}, "
                f"have {n_avail_active}"
            )
            sample_log.append(entry)
            log.debug("Dropped %s - %s", uid, entry["drop_reason"])
            continue

        if n_avail_inactive < n_inactives:
            entry["status"]      = "dropped"
            entry["drop_reason"] = (
                f"insufficient_inactives: need {n_inactives}, "
                f"have {n_avail_inactive}"
            )
            sample_log.append(entry)
            log.debug("Dropped %s - %s", uid, entry["drop_reason"])
            continue

        # â”€â”€ Deterministic sampling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        active_idx   = rng.choice(len(actives_pool),   EXACT_ACTIVES, replace=False)
        inactive_idx = rng.choice(len(inactives_pool), n_inactives,   replace=False)

        sampled = pd.concat([
            actives_pool.iloc[active_idx],
            inactives_pool.iloc[inactive_idx],
        ], ignore_index=True)

        entry["n_sampled_actives"]   = EXACT_ACTIVES
        entry["n_sampled_inactives"] = n_inactives
        sample_log.append(entry)
        sampled_parts.append(sampled)

    final = (
        pd.concat(sampled_parts, ignore_index=True)
        if sampled_parts
        else pd.DataFrame()
    )

    n_retained  = sum(1 for e in sample_log if e["status"] == "retained")
    n_dropped_s = sum(1 for e in sample_log if e["status"] == "dropped")
    log.info(
        "Sampling (%s): %d kinases retained, %d dropped for insufficient data.",
        ratio_mode, n_retained, n_dropped_s,
    )

    return final, sample_log


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  ORCHESTRATOR â€” build_dataset()                                         â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_dataset(
    excel_path:          str | Path  = "ML.xlsx",
    out_dir:             str | Path  = "./pipeline_outputs",
    ratio_mode:          str         = "1:1",
    active_pic50_cutoff: float       = DEFAULT_ACTIVE_CUTOFF_PIC50,
    seed:                int         = 42,
    max_workers:         int         = MAX_WORKERS,
    use_cache:           bool        = True,
) -> pd.DataFrame:
    """
    Full pipeline orchestrator.

    Execution order (ALWAYS protein-first):
        1. Load registry from ML.xlsx
        2. Protein-feasibility filter -> retained targets
        3. Fetch ligands for retained targets only
        4. Standardise + source-priority dedup
        5. Exact per-kinase sampling (ratio_mode)
        6. Persist all artefacts
    """
    t0      = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate ratio_mode
    if ratio_mode not in RATIO_INACTIVES:
        raise ValueError(
            f"Invalid ratio_mode '{ratio_mode}'. "
            f"Allowed: {list(RATIO_INACTIVES.keys())}"
        )
    n_inactives = RATIO_INACTIVES[ratio_mode]

    log.info("=" * 65)
    log.info("Kinase-Ligand Dataset Builder  v2 (protein-feasibility-first)")
    log.info("  ratio_mode     : %s  (%d actives / %d inactives per kinase)",
             ratio_mode, EXACT_ACTIVES, n_inactives)
    log.info("  active_cutoff  : pIC50 >= %.2f", active_pic50_cutoff)
    log.info("  seed           : %d", seed)
    log.info("  out_dir        : %s", out_dir)
    log.info("=" * 65)

    # â”€â”€ STAGE 1: Registry + protein-feasibility filter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    registry = load_registry(excel_path)
    retained_targets, prot_audit = run_protein_feasibility_filter(
        registry, out_dir, workers=max_workers, use_cache=use_cache,
    )

    if retained_targets.empty:
        raise RuntimeError(
            "No targets passed protein feasibility filtering. "
            "Check protein_feature_audit.csv for drop reasons."
        )

    n_retained_targets = len(retained_targets)
    log.info(
        "Stage 1 complete: %d / %d targets retained.",
        n_retained_targets, len(registry),
    )

    # â”€â”€ STAGE 2: Ligand fetching (retained targets only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Quota: fetch enough to support exact sampling + headroom for filtering.
    # Headroom factor = 10Ã— (e.g. need 15 actives â†’ fetch up to 150 candidates).
    headroom      = 10
    fetch_quota   = (EXACT_ACTIVES + n_inactives) * headroom

    raw_cache = out_dir / "raw_ligands.parquet"
    if use_cache and raw_cache.exists():
        log.info("Stage 2: loading raw ligands from cache.")
        raw_all = pd.read_parquet(raw_cache)
    else:
        log.info("=" * 65)
        log.info("STAGE 2 - Fetching ligands for %d retained targets ...", n_retained_targets)
        log.info("=" * 65)

        fetch_logs: list[dict] = []
        all_frames: list[pd.DataFrame] = []

        def _fetch_one(row: pd.Series) -> tuple[pd.DataFrame, dict]:
            uid = row["uniprot_id"]
            log_entry: dict = {
                "uniprot_id":         uid,
                "target":             row["target"],
                "chembl_raw":         0,
                "bindingdb_raw":      0,
                "pubchem_raw":        0,
                "total_raw":          0,
                "reached_chembl_quota": False,
            }

            # Source 1: ChEMBL (primary)
            df_chembl = fetch_chembl_ic50(uid, quota=fetch_quota)
            log_entry["chembl_raw"] = len(df_chembl)
            log_entry["reached_chembl_quota"] = len(df_chembl) >= fetch_quota

            frames = [df_chembl] if not df_chembl.empty else []

            # Source 2: BindingDB (only if ChEMBL is insufficient)
            if len(df_chembl) < fetch_quota // 2:
                df_bdb = fetch_bindingdb_ic50(uid)
                log_entry["bindingdb_raw"] = len(df_bdb)
                if not df_bdb.empty:
                    frames.append(df_bdb)

            # Source 3: PubChem (only if ChEMBL + BindingDB still insufficient)
            total_so_far = sum(len(f) for f in frames)
            if total_so_far < fetch_quota // 4:
                df_pc = fetch_pubchem_ic50(uid)
                log_entry["pubchem_raw"] = len(df_pc)
                if not df_pc.empty:
                    frames.append(df_pc)

            combined = (
                pd.concat(frames, ignore_index=True)
                if frames
                else pd.DataFrame()
            )
            log_entry["total_raw"] = len(combined)
            return combined, log_entry

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_fetch_one, row): row["uniprot_id"]
                for _, row in retained_targets.iterrows()
            }
            for fut in progress_iter(
                as_completed(futures),
                total=len(futures),
                desc="Stage 2 ligand fetch",
            ):
                uid = futures[fut]
                try:
                    df_target, log_entry = fut.result()
                    if not df_target.empty:
                        all_frames.append(df_target)
                    fetch_logs.append(log_entry)
                    log.info(
                        "Fetched %s: ChEMBL=%d, BDB=%d, PC=%d -> total=%d",
                        uid,
                        log_entry["chembl_raw"],
                        log_entry["bindingdb_raw"],
                        log_entry["pubchem_raw"],
                        log_entry["total_raw"],
                    )
                except Exception as exc:
                    log.error("Fetch failed for %s: %s", uid, exc)

        raw_all = (
            pd.concat(all_frames, ignore_index=True)
            if all_frames
            else pd.DataFrame()
        )

        # Persist fetch log and raw ligand cache
        pd.DataFrame(fetch_logs).to_csv(out_dir / "fetch_log.csv", index=False)
        if not raw_all.empty:
            raw_all.to_parquet(raw_cache, index=False)
        log.info("Fetched %d raw records for %d targets.", len(raw_all), n_retained_targets)

    if raw_all.empty:
        raise RuntimeError("No ligands fetched for any retained target.")

    # â”€â”€ STAGE 3: Standardise + source-priority deduplication â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log.info("=" * 65)
    log.info("STAGE 3 - Standardisation + source-priority deduplication ...")
    log.info("=" * 65)

    clean, dup_audit = standardise_and_deduplicate(raw_all, active_pic50_cutoff)

    # Merge protein metadata back in
    clean = clean.merge(
        retained_targets[["uniprot_id", "target", "pdb_id",
                          "protein_status", "feature_status"]],
        on="uniprot_id",
        how="left",
    )

    # â”€â”€ STAGE 4: Exact per-kinase sampling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log.info("=" * 65)
    log.info("STAGE 4 - Exact per-kinase sampling (ratio_mode=%s) ...", ratio_mode)
    log.info("=" * 65)

    final, sample_log = sample_kinase_dataset(clean, ratio_mode, seed)

    # â”€â”€ Final schema enforcement â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    required_cols = [
        "target", "uniprot_id", "pdb_id",
        "inchikey", "canonical_smiles",
        "ic50_nm", "pIC50", "pIC50_std",
        "activity_label", "assay_source", "source_priority",
        "n_measurements", "protein_status", "feature_status",
        "assay_id",
    ]
    for col in required_cols:
        if col not in final.columns:
            final[col] = np.nan

    # Add retained_reason
    final["retained_reason"] = "passed_all_stages"

    # Add smiles alias for backward compatibility
    final["smiles"] = final["canonical_smiles"]

    # â”€â”€ Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    n_kin    = final["uniprot_id"].nunique()
    n_act    = (final["activity_label"] == "active").sum()
    n_inact  = (final["activity_label"] == "inactive").sum()
    n_ligand = final["inchikey"].nunique()
    elapsed  = time.time() - t0

    log.info("=" * 65)
    log.info("FINAL DATASET SUMMARY")
    log.info("  Total records          : %d", len(final))
    log.info("  Kinases retained       : %d / %d", n_kin, n_retained_targets)
    log.info("  Unique ligands         : %d", n_ligand)
    log.info("  Actives                : %d", n_act)
    log.info("  Inactives              : %d", n_inact)
    log.info("  ratio_mode             : %s", ratio_mode)
    log.info("  pIC50 range            : [%.2f, %.2f]",
             final["pIC50"].min(), final["pIC50"].max())
    log.info("  Elapsed                : %.1f s (%.1f min)",
             elapsed, elapsed / 60)
    log.info("=" * 65)

    # â”€â”€ Persist outputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _persist_outputs(
        final, clean, dup_audit, sample_log,
        prot_audit, retained_targets, out_dir, ratio_mode,
    )

    return final


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Persistence helper
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _persist_outputs(
    final:            pd.DataFrame,
    clean_before_samp: pd.DataFrame,
    dup_audit:        pd.DataFrame,
    sample_log:       list[dict],
    prot_audit:       pd.DataFrame,
    retained_targets: pd.DataFrame,
    out_dir:          Path,
    ratio_mode:       str,
) -> None:
    """Write all pipeline artefacts to out_dir."""

    # Main dataset
    out_parquet = out_dir / "dataset_clean.parquet"
    final.to_parquet(out_parquet, index=False)
    log.info("Saved -> %s", out_parquet)

    # Actives / inactives CSVs
    actives   = final[final["activity_label"] == "active"]
    inactives = final[final["activity_label"] == "inactive"]
    actives.to_csv(out_dir / "actives.csv",   index=False)
    inactives.to_csv(out_dir / "inactives.csv", index=False)

    # Dataset summary
    summary_rows = []
    for uid, grp in final.groupby("uniprot_id"):
        pic50_mean = round(float(grp["pIC50"].mean()), 4)
        pic50_std = round(float(grp["pIC50"].std(ddof=0)), 4) if len(grp) > 1 else 0.0
        summary_rows.append({
            "uniprot_id":  uid,
            "target":      grp["target"].iloc[0],
            "pdb_id":      grp["pdb_id"].iloc[0],
            "n_actives":   (grp["activity_label"] == "active").sum(),
            "n_inactives": (grp["activity_label"] == "inactive").sum(),
            "n_total":     len(grp),
            "ratio_mode":  ratio_mode,
            "pIC50_mean":  pic50_mean,
            "pIC50_std":   pic50_std,
            "sources":     ",".join(sorted(grp["assay_source"].unique())),
        })
    pd.DataFrame(summary_rows).to_csv(out_dir / "dataset_summary.csv", index=False)

    # Duplicate audit
    dup_audit.to_csv(out_dir / "duplicate_audit.csv", index=False)

    # Source provenance
    prov = (
        clean_before_samp.groupby(["uniprot_id", "assay_source"])
        .size()
        .reset_index(name="n_records")
    )
    prov.to_csv(out_dir / "source_provenance.csv", index=False)

    # Sampling log
    pd.DataFrame(sample_log).to_csv(out_dir / "sampling_log.csv", index=False)

    log.info(
        "Artefacts saved: dataset_clean.parquet, actives.csv, inactives.csv, "
        "dataset_summary.csv, duplicate_audit.csv, source_provenance.csv, "
        "sampling_log.csv"
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Entry point
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Build a balanced IC50 kinaseâ€“ligand dataset. "
            "Protein feasibility is filtered FIRST; ligands are fetched only "
            "for retained targets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--excel", default="ML.xlsx",
        help="Input kinase registry Excel file (must have: target, uniprot_id, pdb_id)",
    )
    parser.add_argument(
        "--out_dir", default="./pipeline_outputs",
        help="Directory for all pipeline outputs",
    )
    parser.add_argument(
        "--ratio_mode", default="1:1", choices=["1:1", "1:2", "1:3"],
        help=(
            "Active:inactive sampling ratio per kinase. "
            "1:1 â†’ 15+15, 1:2 â†’ 15+30, 1:3 â†’ 15+45"
        ),
    )
    parser.add_argument(
        "--active_pic50_cutoff", type=float, default=DEFAULT_ACTIVE_CUTOFF_PIC50,
        help="pIC50 threshold for active/inactive classification",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic sampling",
    )
    parser.add_argument(
        "--workers", type=int, default=MAX_WORKERS,
        help="Parallel fetch threads",
    )
    parser.add_argument(
        "--no_cache", action="store_true",
        help="Ignore all caches and re-run from scratch",
    )
    args = parser.parse_args()

    df = build_dataset(
        excel_path          = args.excel,
        out_dir             = args.out_dir,
        ratio_mode          = args.ratio_mode,
        active_pic50_cutoff = args.active_pic50_cutoff,
        seed                = args.seed,
        max_workers         = args.workers,
        use_cache           = not args.no_cache,
    )

    print(f"\n{'='*60}")
    print(f"Dataset built       : {len(df):,} records")
    print(f"Kinases             : {df['uniprot_id'].nunique()}")
    print(f"Unique ligands      : {df['inchikey'].nunique():,}")
    print(f"Actives             : {(df['activity_label']=='active').sum():,}")
    print(f"Inactives           : {(df['activity_label']=='inactive').sum():,}")
    print(f"ratio_mode          : {args.ratio_mode}")
    print(f"pIC50 range         : [{df['pIC50'].min():.2f}, {df['pIC50'].max():.2f}]")
    print(f"{'='*60}\n")
    print(df.head(10).to_string(index=False))

