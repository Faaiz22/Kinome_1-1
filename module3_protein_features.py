"""
module3_protein_features.py
===========================
TITLE
Protein feature feasibility and pocket feature construction.

PURPOSE
This module decides which kinase targets are usable and builds the retained
protein feature store consumed by every downstream modelling stage.

WHAT IT DOES
- Validates UniProt/KLIFS feasibility.
- Fetches sequence and structural context when available.
- Produces aligned pocket tensors for retained kinases.
- Saves and reloads the protein feature store.

HOW IT WORKS
1. Validate target feasibility with UniProt and KLIFS checks.
2. Fetch sequence and optional AlphaFold structure support.
3. Build aligned pocket embeddings, coordinates, confidence, and masks.
4. Persist one `ProteinFeatures` object per UniProt ID.

INPUT CONTRACT
- Retained target table containing `uniprot_id`.
- Optional external API availability when cache is absent.

OUTPUT CONTRACT
- `ProteinFeatures` objects keyed by UniProt ID.
- `retained_targets.csv`, audit logs, and protein feature store files.

DEPENDENCIES
- pandas, numpy, torch, requests
- optional ESM and KLIFS/OpenCADD support

CRITICAL ASSUMPTIONS
- KLIFS pocket alignment is the canonical protein representation.
- Targets without pocket feasibility should not proceed downstream.

FAILURE MODES
- Invalid or missing UniProt IDs
- Missing KLIFS mapping
- API/network failures
- Partial structure availability

SAFETY CHECKS IMPLEMENTED
- Explicit drop reasons
- Cache-aware loading
- Fallback behaviour for missing structure confidence inputs

HOW TO RUN
- `python module3_protein_features.py --retained ./pipeline_outputs/retained_targets.csv --output ./pipeline_outputs/protein_feature_store.pt`

HOW IT CONNECTS TO PIPELINE
It defines the retained target universe and the protein tensors used by
training, evaluation, uncertainty estimation, and the Streamlit app.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
import torch
from progress_utils import progress_iter

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# .env loader
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

_load_dotenv()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Configuration
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ALPHAFOLD_API_BASE = os.environ.get("ALPHAFOLD_API_BASE", "https://alphafold.ebi.ac.uk/api")
ESM_MODEL_NAME     = os.environ.get("ESM_MODEL_NAME",     "esm2_t33_650M_UR50D")
ESM_MODEL_CACHE    = os.environ.get("ESM_MODEL_CACHE",    str(Path.home() / ".cache" / "esm"))
KLIFS_LOCAL_CACHE  = os.environ.get("KLIFS_LOCAL_CACHE",  "./klifs_cache")
UNIPROT_API_BASE   = os.environ.get("UNIPROT_API_BASE",   "https://rest.uniprot.org")
UNIPROT_LEGACY_API_BASE = os.environ.get("UNIPROT_LEGACY_API_BASE", "https://www.uniprot.org/uniprot")

KLIFS_POCKET_SIZE: int   = 85
ESM_EMBED_DIM:     int   = 1280
REQUEST_RETRIES:   int   = 5
REQUEST_BACKOFF:   float = 2.0
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
POCKET_DFG_INDEX_1BASED: int = 81
HTTP_CONNECT_TIMEOUT: float = 10.0
HTTP_READ_TIMEOUT: float = 30.0
HTTP_HEADERS = {
    "User-Agent": "kinase-drug-discovery/2.0",
    "Accept": "application/json, text/plain, */*",
}
KLIFS_BASE_URLS: tuple[str, ...] = (
    "https://klifs.net",
    "https://www.klifs.net",
)
KINASE_MIN_LENGTH: int = 200
KINASE_POCKET_MIN_REAL_RESIDUES: int = 70
KINASE_POCKET_MIN_CONFIDENCE: float = 0.5
KINASE_POCKET_DEBUG: bool = os.environ.get("KINASE_POCKET_DEBUG", "").strip().lower() in {
    "1", "true", "yes", "on"
}

log = logging.getLogger("module3")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

_HTTP = requests.Session()
_HTTP.headers.update(HTTP_HEADERS)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Data container
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@dataclass
class ProteinFeatures:
    """
    All protein features for a single kinase pocket.

    All tensors are exactly (85, â€¦) or (85,) â€” padded with zeros for
    missing residues.  pocket_mask indicates real vs padded positions.
    """
    uniprot_id:    str
    esm_pocket:    torch.Tensor          # (85, 1280)
    coords:        torch.Tensor          # (85, 3)
    plddt:         torch.Tensor          # (85,)
    confidence:    torch.Tensor          # (85,)  = (plddt / 100) * pocket_confidence
    pocket_mask:   torch.Tensor          # (85,)  bool
    klifs_indices: list[int | None]      = field(default_factory=list)
    full_sequence: str                   = ""
    has_structure: bool                  = False
    has_klifs:     bool                  = False
    pocket_confidence: float             = 1.0
    pocket_method: str                   = "klifs"
    pocket_warnings: list[str]           = field(default_factory=list)

    def to_device(self, device: str | torch.device) -> "ProteinFeatures":
        self.esm_pocket  = self.esm_pocket.to(device)
        self.coords      = self.coords.to(device)
        self.plddt       = self.plddt.to(device)
        self.confidence  = self.confidence.to(device)
        self.pocket_mask = self.pocket_mask.to(device)
        return self


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# HTTP utility
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class HttpStatusError(RuntimeError):
    """Raised for terminal HTTP status codes that should not be retried."""


class HttpTransportError(RuntimeError):
    """Raised for transient transport failures after exhausting retries."""


def _get_with_retry(url: str, params: Optional[dict] = None, **kw) -> requests.Response:
    delay = REQUEST_BACKOFF
    timeout = kw.pop("timeout", (HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT))
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            resp = _HTTP.get(url, params=params, timeout=timeout, **kw)
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                raise HttpStatusError(f"HTTP {resp.status_code} for {url}")
            resp.raise_for_status()
            return resp
        except HttpStatusError:
            raise
        except requests.RequestException as exc:
            if attempt == REQUEST_RETRIES:
                raise HttpTransportError(
                    f"HTTP GET failed after {REQUEST_RETRIES} attempts [{url}]: {exc}"
                ) from exc
            log.debug(
                "HTTP attempt %d/%d failed for %s; retrying in %.1fs",
                attempt,
                REQUEST_RETRIES,
                url,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("Unreachable")


def _fetch_json_from_candidates(
    urls: list[str],
    *,
    params: Optional[dict] = None,
) -> Optional[object]:
    transport_errors: list[str] = []
    for url in urls:
        try:
            resp = _get_with_retry(url, params=params)
        except HttpStatusError:
            continue
        except HttpTransportError as exc:
            transport_errors.append(str(exc))
            continue
        try:
            return resp.json()
        except ValueError:
            continue
    if transport_errors:
        raise HttpTransportError(" | ".join(transport_errors))
    return None


def _extract_first(
    record: dict,
    keys: tuple[str, ...],
    default: Optional[object] = None,
) -> Optional[object]:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def _target_name_candidates(target_name: Optional[str]) -> list[str]:
    if not target_name or not isinstance(target_name, str):
        return []

    raw = target_name.strip()
    if not raw:
        return []

    candidates: list[str] = []

    def _add(value: str) -> None:
        cleaned = value.strip()
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    _add(raw)

    paren_parts = re.findall(r"\(([^()]+)\)", raw)
    for part in paren_parts:
        _add(part)

    base = re.sub(r"\([^()]*\)", "", raw).strip()
    _add(base)

    sanitized = re.sub(r"[^A-Za-z0-9]+", " ", raw).strip()
    _add(sanitized)

    return candidates


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def infer_kinase_pocket(sequence: str) -> dict[str, Any] | None:
    """
    Infer a conservative KLIFS-like pocket approximation from sequence alone.

    This fallback is intentionally strict. It should only be used when KLIFS
    lookup fails, and it returns None whenever motif consistency is too weak to
    defend biologically.
    """
    if not isinstance(sequence, str):
        return None

    seq = re.sub(r"\s+", "", sequence).upper()
    if len(seq) < KINASE_MIN_LENGTH:
        return None
    if not re.fullmatch(r"[A-Z]+", seq):
        return None

    warnings: list[str] = []
    confidence = 1.0

    gly_loop_matches = list(re.finditer(r"G.G..G", seq[:120]))
    vaik_matches = list(re.finditer(r"[VILM][A-Z]{2}K", seq[20:140]))
    if not gly_loop_matches and not vaik_matches:
        warnings.append("No kinase-like glycine-rich loop or VAIK-like catalytic lysine motif detected.")
        confidence -= 0.35
    elif not gly_loop_matches or not vaik_matches:
        warnings.append("Only one kinase-like catalytic signature was detected.")
        confidence -= 0.15

    dfg_hits = list(re.finditer(r"D[FLY]G", seq))
    if not dfg_hits:
        return None

    hrd_hits = list(re.finditer(r"H[RG]D", seq))
    if not hrd_hits:
        return None

    if len(dfg_hits) > 1:
        warnings.append(f"Multiple DFG-like motifs detected ({len(dfg_hits)}).")
        confidence -= 0.15

    pair_candidates: list[dict[str, Any]] = []
    for dfg_hit in dfg_hits:
        dfg_index = dfg_hit.start() + 1  # 1-based
        upstream_hrd = [h for h in hrd_hits if h.start() < dfg_hit.start()]
        if not upstream_hrd:
            continue
        hrd_hit = min(upstream_hrd, key=lambda h: abs((dfg_hit.start() + 1) - (h.start() + 1) - 30))
        hrd_index = hrd_hit.start() + 1
        distance = dfg_index - hrd_index
        valid = 20 <= distance <= 40
        pair_candidates.append(
            {
                "dfg_index": dfg_index,
                "hrd_index": hrd_index,
                "distance": distance,
                "valid": valid,
                "dfg_motif": dfg_hit.group(),
                "hrd_motif": hrd_hit.group(),
            }
        )

    if not pair_candidates:
        return None

    valid_pairs = [p for p in pair_candidates if p["valid"]]
    if valid_pairs:
        best_pair = min(valid_pairs, key=lambda p: abs(p["distance"] - 30))
        if len(valid_pairs) > 1:
            warnings.append(f"Multiple valid HRD-DFG pairings detected ({len(valid_pairs)}); selected the closest to 30 residues spacing.")
            confidence -= 0.10
    else:
        best_pair = min(pair_candidates, key=lambda p: abs(p["distance"] - 30))
        warnings.append(
            f"HRD-DFG spacing outside the ideal 20-40 residue range (observed {best_pair['distance']})."
        )
        confidence -= 0.25

    dfg_index = int(best_pair["dfg_index"])
    hrd_index = int(best_pair["hrd_index"])
    distance = int(best_pair["distance"])

    # Approximate contiguous fallback window: substantial upstream kinase domain
    # context plus a short downstream segment around the DFG motif.
    window_start = dfg_index - 69
    window_end = window_start + KLIFS_POCKET_SIZE - 1
    positions: list[int | None] = []
    for seq_pos in range(window_start, window_end + 1):
        if 1 <= seq_pos <= len(seq):
            positions.append(seq_pos)
        else:
            positions.append(None)

    n_real = sum(pos is not None for pos in positions)
    if n_real < KINASE_POCKET_MIN_REAL_RESIDUES:
        return None

    if not gly_loop_matches:
        warnings.append("No glycine-rich loop motif GxGxxG detected in the N-lobe region.")
    if not vaik_matches:
        warnings.append("No VAIK-like catalytic lysine motif detected in the N-lobe region.")
    if not gly_loop_matches and not vaik_matches and not valid_pairs:
        return None

    coverage_penalty = 0.30 * ((KLIFS_POCKET_SIZE - n_real) / max(1, KLIFS_POCKET_SIZE - KINASE_POCKET_MIN_REAL_RESIDUES))
    confidence -= max(0.0, coverage_penalty)
    confidence = max(0.0, min(1.0, confidence))

    if KINASE_POCKET_DEBUG:
        log.debug(
            "Sequence fallback motifs: len=%d gly=%s vaik=%s dfg=%s hrd=%s pair=%s",
            len(seq),
            [m.start() + 1 for m in gly_loop_matches],
            [m.start() + 21 for m in vaik_matches],
            [m.start() + 1 for m in dfg_hits],
            [m.start() + 1 for m in hrd_hits],
            best_pair,
        )

    if confidence < KINASE_POCKET_MIN_CONFIDENCE:
        return None

    return {
        "positions": positions,
        "dfg_index": dfg_index,
        "hrd_index": hrd_index,
        "confidence": confidence,
        "method": "sequence_fallback",
        "warnings": warnings,
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  PUBLIC API â€” Protein Feasibility Gate                                  â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def validate_protein_feasibility(
    uniprot_id: str,
    klifs_mapper: "KLIFSPocketMapper",
    target_name: Optional[str] = None,
    pdb_id: Optional[str] = None,
) -> dict:
    """
    Check whether proper protein features can be built for a UniProt ID.

    This is the FIRST gate in the pipeline.  module1 calls this before any
    ligand fetching.

    Checks performed (in order):
        1. UniProt ID format validation (regex)
        2. UniProt sequence retrieval
        3. KLIFS pocket mapping availability

    AlphaFold availability is NOT a hard requirement â€” coordinates fall back
    to zeros when the structure is unavailable.

    Returns
    -------
    dict with keys:
        uniprot_id       : str
        protein_status   : 'retained' | 'dropped'
        feature_status   : 'pending' | 'failed'
        drop_reason      : str (empty if retained)
        seq_length       : int
        klifs_residues   : int (non-None positions in KLIFS mapping)
        alphafold_ok     : bool
    """
    entry: dict = {
        "uniprot_id":      uniprot_id,
        "protein_status":  "dropped",
        "feature_status":  "failed",
        "drop_reason":     "",
        "seq_length":      0,
        "klifs_residues":  0,
        "alphafold_ok":    False,
        "pocket_confidence": 0.0,
        "pocket_method":   "",
    }

    # â”€â”€ 1. UniProt ID format â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not re.match(r"^[A-Z0-9]{6,10}$", uniprot_id.strip()):
        entry["drop_reason"] = "invalid_uniprot_id_format"
        return entry

    # â”€â”€ 2. Sequence retrieval â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    seq = fetch_uniprot_sequence(uniprot_id)
    if seq is None:
        entry["drop_reason"] = "uniprot_sequence_not_found"
        return entry
    entry["seq_length"] = len(seq)

    # â”€â”€ 3. AlphaFold check (non-fatal) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    af_ok = check_alphafold_available(uniprot_id)
    entry["alphafold_ok"] = af_ok
    if not af_ok:
        log.debug("%s: AlphaFold unavailable - zero-coord fallback will be used.", uniprot_id)

    # â”€â”€ 4. KLIFS pocket mapping â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        pocket_def = klifs_mapper.get_pocket_definition(
            uniprot_id,
            sequence=seq,
            target_name=target_name,
            pdb_id=pdb_id,
        )
    except Exception as exc:
        log.warning("KLIFS lookup raised for %s: %s", uniprot_id, exc)
        pocket_def = None

    if pocket_def is None:
        entry["drop_reason"] = "no_klifs_pocket_mapping"
        return entry

    indices = pocket_def["positions"]
    entry["pocket_confidence"] = float(pocket_def.get("confidence", 0.0))
    entry["pocket_method"] = str(pocket_def.get("method", ""))

    n_real = sum(1 for x in indices if x is not None)
    if n_real == 0:
        entry["drop_reason"] = "klifs_pocket_all_missing_residues"
        return entry

    entry["klifs_residues"]  = n_real
    entry["protein_status"]  = "retained"
    entry["feature_status"]  = "pending"   # full ESM-2 done in build_feature_store()
    return entry


def run_feasibility_filter_stage(
    registry:    pd.DataFrame,
    out_dir:     Path,
    workers:     int  = 8,
    use_cache:   bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run protein feasibility filtering for all targets in the registry.

    Called by module1.build_dataset() as Stage 1.

    Returns
    -------
    retained_targets : DataFrame  (subset of registry that passed)
    audit            : DataFrame  (full audit â€” all targets)
    """
    audit_cache  = out_dir / "protein_feature_audit.csv"
    retain_cache = out_dir / "retained_targets.csv"

    if use_cache and audit_cache.exists() and retain_cache.exists():
        log.info("Stage 1: protein feasibility audit loaded from cache.")
        audit = pd.read_csv(audit_cache)
        retained = pd.read_csv(retain_cache)
        if retained.empty and len(registry) > 0:
            log.warning(
                "Cached retained_targets.csv is empty for a non-empty registry; "
                "ignoring stale cache and rebuilding Stage 1 feasibility."
            )
        else:
            log.info("  Retained: %d / %d", len(retained), len(registry))
            return retained, audit

    log.info("=" * 65)
    log.info("STAGE 1 - Protein feasibility filter (%d targets)", len(registry))
    log.info("=" * 65)

    klifs_mapper = KLIFSPocketMapper()
    audit_rows: list[dict] = []

    def _check(row: pd.Series) -> dict:
        result = validate_protein_feasibility(
            row["uniprot_id"],
            klifs_mapper,
            target_name=row.get("target", ""),
            pdb_id=row.get("pdb_id", ""),
        )
        result["target"] = row.get("target", "")
        result["pdb_id"] = row.get("pdb_id", "")
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_check, row): i for i, row in registry.iterrows()}
        for fut in progress_iter(
            as_completed(futures),
            total=len(futures),
            desc="Stage 1 feasibility",
        ):
            try:
                audit_rows.append(fut.result())
            except Exception as exc:
                log.error("Feasibility check raised: %s", exc)

    audit = pd.DataFrame(audit_rows)

    # Merge back with registry for full retained DataFrame
    retained = registry.merge(
        audit[audit["protein_status"] == "retained"][
            ["uniprot_id", "protein_status", "feature_status",
             "seq_length", "klifs_residues", "alphafold_ok",
             "pocket_confidence", "pocket_method"]
        ],
        on="uniprot_id", how="inner",
    ).reset_index(drop=True)

    n_total    = len(registry)
    n_retained = len(retained)
    n_dropped  = n_total - n_retained
    if n_retained == 0 and klifs_mapper._rest_disable_reason:
        raise RuntimeError(
            "KLIFS API was unreachable during Stage 1, so no targets could be retained. "
            f"Reason: {klifs_mapper._rest_disable_reason}"
        )
    log.info("Stage 1 complete - retained: %d / %d  (dropped: %d)",
             n_retained, n_total, n_dropped)

    if n_dropped:
        reasons = audit[audit["protein_status"] != "retained"]["drop_reason"].value_counts()
        for reason, count in reasons.items():
            log.info("  %-40s : %d", reason, count)

    # Persist artefacts
    out_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_cache, index=False)
    retained.to_csv(retain_cache, index=False)

    registry_clean = registry.merge(
        audit[["uniprot_id", "protein_status", "feature_status",
               "drop_reason", "seq_length", "klifs_residues", "alphafold_ok",
               "pocket_confidence", "pocket_method"]],
        on="uniprot_id", how="left",
    )
    registry_clean.to_parquet(out_dir / "registry_clean.parquet", index=False)
    log.info("Saved -> registry_clean.parquet, protein_feature_audit.csv, retained_targets.csv")

    if not audit.empty and "pocket_confidence" in audit.columns:
        fallback_conf = audit.loc[audit["pocket_method"] == "sequence_fallback", "pocket_confidence"].dropna()
        if not fallback_conf.empty:
            log.info(
                "Sequence fallback pocket confidence: n=%d mean=%.3f min=%.3f max=%.3f",
                len(fallback_conf),
                float(fallback_conf.mean()),
                float(fallback_conf.min()),
                float(fallback_conf.max()),
            )

    return retained, audit


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# UniProt
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _parse_fasta_sequence(fasta_text: str, uniprot_id: str) -> Optional[str]:
    lines = [line.strip() for line in fasta_text.splitlines() if line.strip()]
    if not lines or not lines[0].startswith(">"):
        log.warning("UniProt returned a non-FASTA payload for %s", uniprot_id)
        return None

    seq = "".join(lines[1:]).upper()
    if not seq:
        log.warning("UniProt returned an empty sequence for %s", uniprot_id)
        return None
    if re.search(r"[^A-Z\*]", seq):
        log.warning("UniProt returned invalid FASTA characters for %s", uniprot_id)
        return None
    return seq

def fetch_uniprot_sequence(uniprot_id: str) -> Optional[str]:
    """Fetch the canonical amino-acid sequence from UniProt endpoints."""
    urls = [
        f"{UNIPROT_API_BASE}/uniprotkb/{uniprot_id}.fasta",
        f"{UNIPROT_LEGACY_API_BASE}/{uniprot_id}.fasta",
    ]
    for url in urls:
        try:
            resp = _get_with_retry(url, headers={"Accept": "text/x-fasta"})
            seq = _parse_fasta_sequence(resp.text, uniprot_id)
            if seq:
                return seq
        except RuntimeError as exc:
            log.error("Sequence fetch failed for %s via %s: %s", uniprot_id, url, exc)
    return None


def check_alphafold_available(uniprot_id: str) -> bool:
    """Return True if AlphaFold has a structure for this UniProt ID."""
    url = f"{ALPHAFOLD_API_BASE}/prediction/{uniprot_id}"
    try:
        resp = requests.get(url, timeout=20)
        return resp.status_code == 200 and bool(resp.json())
    except Exception:
        return False


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ESM-2 embedder (lazy singleton)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ESM2Embedder:
    """
    Lazy-loading singleton wrapper around ESM-2.

    Sequences > 1022 residues are handled by extracting a kinase-domain
    window around the KLIFS pocket before embedding.
    """

    _instance:         Optional["ESM2Embedder"] = None
    _model                                       = None
    _alphabet                                    = None
    _batch_converter                             = None
    _load_error:       Optional[str]             = None

    MAX_LEN = 1022   # ESM-2 hard context window

    @classmethod
    def get_instance(cls) -> "ESM2Embedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self) -> None:
        if self._model is not None or self._load_error is not None:
            return
        log.info("Loading ESM-2 (%s) ...", ESM_MODEL_NAME)
        try:
            import esm
            torch.hub.set_dir(ESM_MODEL_CACHE)
            model, alphabet = esm.pretrained.__dict__[ESM_MODEL_NAME]()
            model = model.eval().to(DEVICE)
            self.__class__._model           = model
            self.__class__._alphabet        = alphabet
            self.__class__._batch_converter = alphabet.get_batch_converter()
            log.info("ESM-2 loaded on %s", DEVICE)
        except (ImportError, KeyError) as exc:
            self.__class__._load_error = str(exc)
            log.warning("ESM-2 unavailable: %s", exc)

    def embed_sequence(
        self,
        uniprot_id:  str,
        sequence:    str,
        klifs_indices: list[int | None],
        repr_layer:  int = 33,
    ) -> Optional[torch.Tensor]:
        """
        Embed and return the (85, 1280) pocket embedding.

        If sequence > 1022 aa, a window around the KLIFS pocket is extracted.

        Returns None on fatal error.
        """
        self._load()
        if self._load_error is not None or self._model is None:
            return None

        # â”€â”€ Sequence windowing for long kinases â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        real_indices = [x for x in klifs_indices if x is not None]
        if not real_indices:
            log.warning("No real KLIFS indices for %s - cannot embed.", uniprot_id)
            return None

        # Pocket centre in 1-based UniProt numbering
        pocket_centre = int(np.median(real_indices))

        if len(sequence) <= self.MAX_LEN:
            sub_seq   = sequence
            seq_offset = 0   # all UniProt positions intact
        else:
            half   = self.MAX_LEN // 2
            start  = max(0, pocket_centre - half - 1)
            end    = min(len(sequence), start + self.MAX_LEN)
            start  = max(0, end - self.MAX_LEN)  # clamp
            sub_seq    = sequence[start:end]
            seq_offset = start
            log.debug(
                "%s: long sequence (%d aa) - windowed to [%d, %d]",
                uniprot_id, len(sequence), start, end,
            )

        data = [(uniprot_id, sub_seq)]
        try:
            _, _, batch_tokens = self._batch_converter(data)
            batch_tokens = batch_tokens.to(DEVICE)

            with torch.no_grad():
                results = self._model(
                    batch_tokens,
                    repr_layers=[repr_layer],
                    return_contacts=False,
                )
            # Shape: (1, L+2, ESM_EMBED_DIM) â†’ remove BOS/EOS
            full_emb = results["representations"][repr_layer][0, 1:-1].cpu()
            assert full_emb.shape == (len(sub_seq), ESM_EMBED_DIM), (
                f"ESM-2 embedding shape mismatch for {uniprot_id}: {full_emb.shape}"
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                log.error("OOM embedding %s - returning None.", uniprot_id)
                return None
            raise

        # â”€â”€ Slice to KLIFS pocket â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        pocket_emb = torch.zeros(KLIFS_POCKET_SIZE, ESM_EMBED_DIM)
        for i, res_num in enumerate(klifs_indices):
            if res_num is None:
                continue
            local_idx = res_num - 1 - seq_offset   # convert to window-local 0-based
            if 0 <= local_idx < len(sub_seq):
                pocket_emb[i] = full_emb[local_idx]

        return pocket_emb   # (85, 1280)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# KLIFS pocket mapper (cached, with opencadd + REST fallback)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class KLIFSPocketMapper:
    """
    Retrieve the 85-residue KLIFS pocket mapping for a UniProt ID.

    Cache: JSON files in KLIFS_LOCAL_CACHE per UniProt ID.
    Fallback chain: opencadd -> KLIFS REST URL variants -> cached result.
    """

    def __init__(self, cache_dir: str = KLIFS_LOCAL_CACHE) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._use_opencadd = self._check_opencadd()
        self._kinase_catalog: Optional[list[dict]] = None
        self._lock = threading.Lock()
        self._rest_disabled = False
        self._rest_disable_reason = ""

    @staticmethod
    def _check_opencadd() -> bool:
        try:
            import opencadd.databases.klifs  # noqa: F401
            return True
        except ImportError:
            log.info("opencadd not available - using KLIFS REST API.")
            return False

    def _disable_rest(self, reason: str) -> None:
        with self._lock:
            if self._rest_disabled:
                return
            self._rest_disabled = True
            self._rest_disable_reason = reason
        log.warning("KLIFS REST temporarily disabled for this run: %s", reason)

    @staticmethod
    def _normalize_pdb_id(pdb_id: Optional[str]) -> str:
        if pdb_id is None:
            return ""
        text = str(pdb_id).strip().upper()
        return text if re.fullmatch(r"[0-9A-Z]{4}", text) else ""

    @staticmethod
    def _select_best_structure_id(records: object) -> Optional[int]:
        if not isinstance(records, list):
            return None

        best_struct_id: Optional[int] = None
        best_missing = float("inf")
        for record in records:
            if not isinstance(record, dict):
                continue
            struct_id = _extract_first(record, ("structure_ID", "structure_id", "id"))
            if struct_id is None:
                continue
            try:
                struct_id_int = int(struct_id)
            except (TypeError, ValueError):
                continue

            missing = _extract_first(record, ("missing_residues", "structure.missing_residues"), float("inf"))
            if isinstance(missing, (int, float)) and missing < best_missing:
                best_missing = float(missing)
                best_struct_id = struct_id_int

        return best_struct_id

    def _fetch_pocket_by_structure_id(self, structure_id: int) -> Optional[list[int | None]]:
        pocket_urls = [
            f"{base}/api_v2/interactions/pockets/{structure_id}"
            for base in KLIFS_BASE_URLS
        ] + [
            f"{base}/api/interactions/pockets/{structure_id}"
            for base in KLIFS_BASE_URLS
        ]
        pocket_data = _fetch_json_from_candidates(pocket_urls)
        if not pocket_data or len(pocket_data) != KLIFS_POCKET_SIZE:
            return None

        result: list[int | None] = []
        for entry in pocket_data:
            rn = _extract_first(entry, ("residue.id", "residue_id", "residue_number"))
            if rn is None:
                result.append(None)
                continue
            try:
                result.append(int(rn))
            except (TypeError, ValueError):
                result.append(None)
        return result if any(x is not None for x in result) else None

    def _fetch_via_pdb(self, pdb_id: str) -> Optional[list[int | None]]:
        normalized_pdb = self._normalize_pdb_id(pdb_id)
        if not normalized_pdb:
            return None

        structure_urls = []
        for base in KLIFS_BASE_URLS:
            structure_urls.extend(
                [
                    f"{base}/api_v2/structures?pdb={normalized_pdb}",
                    f"{base}/api_v2/structures?pdb_ID={normalized_pdb}",
                    f"{base}/api_v2/structures?pdb_id={normalized_pdb}",
                    f"{base}/api_v2/structures?pdb_code={normalized_pdb}",
                    f"{base}/api_v2/structures?pdbCode={normalized_pdb}",
                    f"{base}/api_v2/structures?pdb-codes={normalized_pdb}",
                    f"{base}/api_v2/structures?structure_PDB={normalized_pdb}",
                    f"{base}/api_v2/structures?structure_pdb={normalized_pdb}",
                    f"{base}/api/structures?pdb={normalized_pdb}",
                    f"{base}/api/structures?pdb_ID={normalized_pdb}",
                    f"{base}/api/structures?pdb_id={normalized_pdb}",
                    f"{base}/api/structures?pdb_code={normalized_pdb}",
                    f"{base}/api/structures?pdbCode={normalized_pdb}",
                    f"{base}/api/structures?pdb-codes={normalized_pdb}",
                    f"{base}/api/structures?structure_PDB={normalized_pdb}",
                    f"{base}/api/structures?structure_pdb={normalized_pdb}",
                ]
            )

        records = _fetch_json_from_candidates(_dedupe_preserve_order(structure_urls))
        struct_id = self._select_best_structure_id(records)
        if struct_id is None:
            return None
        return self._fetch_pocket_by_structure_id(struct_id)

    def _fetch_via_target_name(self, target_name: str) -> Optional[list[int | None]]:
        if not target_name:
            return None

        kinase_urls = []
        for base in KLIFS_BASE_URLS:
            kinase_urls.extend(
                [
                    f"{base}/api_v2/kinases?kinase_name={target_name}",
                    f"{base}/api_v2/kinases?name={target_name}",
                    f"{base}/api_v2/kinases?kinase={target_name}",
                    f"{base}/api_v2/kinases?kinase_group={target_name}",
                    f"{base}/api_v2/kinases?gene={target_name}",
                    f"{base}/api/kinases?kinase_name={target_name}",
                    f"{base}/api/kinases?name={target_name}",
                    f"{base}/api/kinases?kinase={target_name}",
                    f"{base}/api/kinases?gene={target_name}",
                ]
            )

        records = _fetch_json_from_candidates(_dedupe_preserve_order(kinase_urls))
        if not isinstance(records, list) or not records:
            return None

        kinase_ids: list[int] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            kinase_id = _extract_first(record, ("kinase_ID", "kinase_id", "id"))
            if kinase_id is None:
                continue
            try:
                kinase_ids.append(int(kinase_id))
            except (TypeError, ValueError):
                continue

        if not kinase_ids:
            return None

        best_struct_id: Optional[int] = None
        best_missing = float("inf")
        for kid in kinase_ids:
            structure_urls = [
                f"{base}/api_v2/structures?kinase_ID={kid}"
                for base in KLIFS_BASE_URLS
            ] + [
                f"{base}/api_v2/structures?kinase_id={kid}"
                for base in KLIFS_BASE_URLS
            ] + [
                f"{base}/api_v2/structures?id={kid}"
                for base in KLIFS_BASE_URLS
            ] + [
                f"{base}/api/structures?kinase_ID={kid}"
                for base in KLIFS_BASE_URLS
            ] + [
                f"{base}/api/structures?kinase_id={kid}"
                for base in KLIFS_BASE_URLS
            ] + [
                f"{base}/api/structures?id={kid}"
                for base in KLIFS_BASE_URLS
            ]
            records = _fetch_json_from_candidates(_dedupe_preserve_order(structure_urls))
            struct_id = self._select_best_structure_id(records)
            if struct_id is None:
                continue
            if isinstance(records, list):
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    maybe_id = _extract_first(record, ("structure_ID", "structure_id", "id"))
                    if maybe_id is None or int(maybe_id) != struct_id:
                        continue
                    missing = _extract_first(record, ("missing_residues", "structure.missing_residues"), float("inf"))
                    if isinstance(missing, (int, float)) and missing < best_missing:
                        best_missing = float(missing)
                        best_struct_id = struct_id

        if best_struct_id is None:
            return None
        return self._fetch_pocket_by_structure_id(best_struct_id)

    def _get_kinase_catalog(self) -> list[dict]:
        if self._kinase_catalog is None:
            catalog_urls = [
                f"{base}/api_v2/kinase_information"
                for base in KLIFS_BASE_URLS
            ] + [
                f"{base}/api/kinase_information"
                for base in KLIFS_BASE_URLS
            ]
            catalog = _fetch_json_from_candidates(_dedupe_preserve_order(catalog_urls))
            if not isinstance(catalog, list):
                raise RuntimeError("KLIFS kinase catalog response is not a list.")
            self._kinase_catalog = catalog
        return self._kinase_catalog

    def _resolve_kinase_id_from_catalog(self, uniprot_id: str) -> Optional[int]:
        try:
            catalog = self._get_kinase_catalog()
        except Exception as exc:
            log.warning("KLIFS catalog fetch failed for %s: %s", uniprot_id, exc)
            return None

        for row in catalog:
            if str(row.get("uniprot", "")).strip().upper() == uniprot_id.upper():
                kinase_id = _extract_first(row, ("kinase_ID", "kinase_id", "id"))
                if kinase_id is None:
                    return None
                try:
                    return int(kinase_id)
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _parse_detail_page_residue_numbers(html: str) -> Optional[list[int | None]]:
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        matches = re.findall(r"(\d{1,2})\s+([A-Z_])\s+(\d{1,5})", text)
        if not matches:
            return None

        pocket_dict: dict[int, int] = {}
        for pos_str, aa, resnum_str in matches:
            try:
                pos = int(pos_str)
                resnum = int(resnum_str)
            except ValueError:
                continue
            if 1 <= pos <= KLIFS_POCKET_SIZE and aa != "_":
                pocket_dict[pos - 1] = resnum

        result = [pocket_dict.get(i) for i in range(KLIFS_POCKET_SIZE)]
        return result if any(x is not None for x in result) else None

    def _fetch_via_catalog_details(self, uniprot_id: str) -> Optional[list[int | None]]:
        kinase_id = self._resolve_kinase_id_from_catalog(uniprot_id)
        if kinase_id is None:
            return None

        structure_urls = [
            f"{base}/api_v2/structures_list?kinase_ID={kinase_id}"
            for base in KLIFS_BASE_URLS
        ] + [
            f"{base}/api/structures_list?kinase_ID={kinase_id}"
            for base in KLIFS_BASE_URLS
        ]
        structures = _fetch_json_from_candidates(_dedupe_preserve_order(structure_urls))
        if not isinstance(structures, list) or not structures:
            return None

        def _sort_key(record: dict) -> tuple[float, float]:
            missing = _extract_first(record, ("missing_residues", "structure.missing_residues"), 9999)
            quality = _extract_first(record, ("quality_score",), 0.0)
            try:
                missing_val = float(missing)
            except (TypeError, ValueError):
                missing_val = 9999.0
            try:
                quality_val = -float(quality)
            except (TypeError, ValueError):
                quality_val = 0.0
            return (missing_val, quality_val)

        structures = sorted([s for s in structures if isinstance(s, dict)], key=_sort_key)
        if not structures:
            return None
        struct_id = _extract_first(structures[0], ("structure_ID", "structure_id", "id"))
        if struct_id is None:
            return None
        try:
            struct_id_int = int(struct_id)
        except (TypeError, ValueError):
            return None

        detail_urls = [
            f"{base}/details.php?structure_id={struct_id_int}"
            for base in KLIFS_BASE_URLS
        ]
        for url in _dedupe_preserve_order(detail_urls):
            try:
                resp = _get_with_retry(url)
            except RuntimeError:
                continue
            result = self._parse_detail_page_residue_numbers(resp.text)
            if result is not None:
                return result
        return None

    def get_pocket_definition(
        self,
        uniprot_id: str,
        sequence: Optional[str] = None,
        target_name: Optional[str] = None,
        pdb_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Return a structured pocket definition for one UniProt accession."""
        cache_file = self.cache_dir / f"{uniprot_id}_klifs.json"
        if cache_file.exists():
            with open(cache_file) as f:
                cached = json.load(f)
            indices = cached.get("positions") or cached.get("pocket_indices")
            if indices is not None:
                cached["positions"] = [None if x is None else int(x) for x in indices]
                cached.setdefault("confidence", 1.0 if cached.get("method", "klifs") == "klifs" else 0.0)
                cached.setdefault("method", "klifs")
                cached.setdefault("warnings", [])
                cached["has_klifs"] = bool(cached.get("has_klifs", cached["method"] == "klifs"))
                cached["from_cache"] = True
                if cached["has_klifs"]:
                    log.info(
                        "%s: KLIFS pocket fetched from cache via %s.",
                        uniprot_id,
                        cached.get("method", "klifs"),
                    )
                return cached

        indices = None
        method = "klifs"
        try:
            if pdb_id:
                indices = self._fetch_via_pdb(str(pdb_id))
                if indices is not None:
                    method = "klifs_pdb"

            if indices is None:
                for candidate in _target_name_candidates(target_name):
                    indices = self._fetch_via_target_name(candidate)
                    if indices is not None:
                        method = "klifs_target"
                        break

            if indices is None and self._use_opencadd:
                indices = self._fetch_via_opencadd(uniprot_id)
                if indices is not None:
                    method = "klifs_opencadd"

            if indices is None:
                indices = self._fetch_via_rest(uniprot_id)
                if indices is not None:
                    method = "klifs_uniprot"

            if indices is None:
                indices = self._fetch_via_catalog_details(uniprot_id)
                if indices is not None:
                    method = "klifs_catalog"
        except HttpTransportError as exc:
            self._disable_rest(f"identifier-based KLIFS lookup transport failure ({uniprot_id}): {exc}")
            indices = None

        pocket_def: Optional[dict[str, Any]]
        if indices is not None:
            pocket_def = {
                "positions": indices,
                "confidence": 1.0,
                "method": method,
                "warnings": [],
                "has_klifs": True,
                "from_cache": False,
            }
            log.info("%s: KLIFS pocket fetched via %s.", uniprot_id, method)
        elif sequence:
            pocket_def = infer_kinase_pocket(sequence)
            if pocket_def is not None:
                pocket_def["has_klifs"] = False
                pocket_def["from_cache"] = False
                log.warning(
                    "%s: using sequence-derived kinase pocket fallback (confidence=%.2f).",
                    uniprot_id,
                    float(pocket_def["confidence"]),
                )
                if pocket_def.get("warnings"):
                    log.warning("%s: fallback warnings: %s", uniprot_id, "; ".join(pocket_def["warnings"]))
        else:
            pocket_def = None

        if pocket_def is not None:
            with open(cache_file, "w") as f:
                json.dump({"uniprot_id": uniprot_id, **pocket_def}, f)

        return pocket_def

    def get_pocket_indices(
        self,
        uniprot_id: str,
        sequence: Optional[str] = None,
        target_name: Optional[str] = None,
        pdb_id: Optional[str] = None,
    ) -> Optional[list[int | None]]:
        pocket_def = self.get_pocket_definition(
            uniprot_id,
            sequence=sequence,
            target_name=target_name,
            pdb_id=pdb_id,
        )
        if pocket_def is None:
            return None
        return pocket_def["positions"]

    # â”€â”€ opencadd path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _fetch_via_opencadd(self, uniprot_id: str) -> Optional[list[int | None]]:
        try:
            from opencadd.databases.klifs import setup_remote
            klifs = setup_remote()
            structures = klifs.structures.by_kinase_name(kinase_name=uniprot_id)
            if structures is None or len(structures) == 0:
                return None
            structures = structures.sort_values("structure.missing_residues")
            struct_id  = int(structures.iloc[0]["structure.klifs_id"])
            pockets    = klifs.pockets.by_structure_klifs_id(struct_id)
            if pockets is None or len(pockets) == 0:
                return None
            result: list[int | None] = []
            for _, row in pockets.iterrows():
                rn = row.get("residue.id")
                result.append(None if (rn is None or str(rn) == "nan") else int(rn))
            if len(result) != KLIFS_POCKET_SIZE:
                log.warning("opencadd KLIFS returned %d residues (expected 85) for %s",
                            len(result), uniprot_id)
                return None
            return result
        except Exception as exc:
            log.debug("opencadd KLIFS failed for %s: %s", uniprot_id, exc)
            return None

    # â”€â”€ REST path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _fetch_via_rest(self, uniprot_id: str) -> Optional[list[int | None]]:
        """
        Query KLIFS using several REST URL variants:
            1. /api_v2|/api/kinases/uniprot?uniprot=<id>  -> kinase IDs
            2. /api_v2|/api/structures?kinase_ID=<id>     -> best structure
            3. /api_v2|/api/interactions/pockets/<id>     -> pocket residues
        """
        if self._rest_disabled:
            return None

        # Step 1: resolve UniProt -> KLIFS kinase_ID
        kinase_lookup_urls = [
            f"{base}/api_v2/kinases/uniprot?uniprot={uniprot_id}"
            for base in KLIFS_BASE_URLS
        ] + [
            f"{base}/api_v2/kinases?uniprot={uniprot_id}"
            for base in KLIFS_BASE_URLS
        ] + [
            f"{base}/api_v2/kinases?uniprot_id={uniprot_id}"
            for base in KLIFS_BASE_URLS
        ] + [
            f"{base}/api/kinases/uniprot?uniprot={uniprot_id}"
            for base in KLIFS_BASE_URLS
        ] + [
            f"{base}/api/kinases?uniprot={uniprot_id}"
            for base in KLIFS_BASE_URLS
        ] + [
            f"{base}/api/kinases?uniprot_id={uniprot_id}"
            for base in KLIFS_BASE_URLS
        ]
        try:
            data = _fetch_json_from_candidates(_dedupe_preserve_order(kinase_lookup_urls))
        except HttpTransportError as exc:
            self._disable_rest(f"kinase lookup transport failure ({uniprot_id}): {exc}")
            return None

        if not data:
            return None

        kinase_ids: list[int] = []
        for record in data:
            kinase_id = _extract_first(record, ("kinase_ID", "kinase_id", "id"))
            if kinase_id is None:
                continue
            try:
                kinase_ids.append(int(kinase_id))
            except (TypeError, ValueError):
                continue
        if not kinase_ids:
            return None

        # Step 2: get best structure for each kinase_ID
        best_struct_id: Optional[int] = None
        best_missing   = float("inf")

        for kid in kinase_ids:
            structure_urls = [
                f"{base}/api_v2/structures?kinase_ID={kid}"
                for base in KLIFS_BASE_URLS
            ] + [
                f"{base}/api_v2/structures?kinase_id={kid}"
                for base in KLIFS_BASE_URLS
            ] + [
                f"{base}/api/structures?kinase_ID={kid}"
                for base in KLIFS_BASE_URLS
            ] + [
                f"{base}/api/structures?kinase_id={kid}"
                for base in KLIFS_BASE_URLS
            ]
            try:
                structs = _fetch_json_from_candidates(_dedupe_preserve_order(structure_urls))
            except HttpTransportError as exc:
                self._disable_rest(f"structures lookup transport failure ({uniprot_id}): {exc}")
                return None
            if not structs:
                continue
            for s in structs:
                missing = _extract_first(s, ("missing_residues", "structure.missing_residues"), float("inf"))
                if isinstance(missing, (int, float)) and missing < best_missing:
                    structure_id = _extract_first(s, ("structure_ID", "structure_id", "id"))
                    if structure_id is None:
                        continue
                    try:
                        best_missing = missing
                        best_struct_id = int(structure_id)
                    except (TypeError, ValueError):
                        continue

        if best_struct_id is None:
            log.warning("No KLIFS structures found for %s", uniprot_id)
            return None

        # Step 3: fetch pocket residues
        pocket_urls = [
            f"{base}/api_v2/interactions/pockets/{best_struct_id}"
            for base in KLIFS_BASE_URLS
        ] + [
            f"{base}/api_v2/structure_get_pocket?structure_ID={best_struct_id}"
            for base in KLIFS_BASE_URLS
        ] + [
            f"{base}/api/interactions/pockets/{best_struct_id}"
            for base in KLIFS_BASE_URLS
        ] + [
            f"{base}/api/structure_get_pocket?structure_ID={best_struct_id}"
            for base in KLIFS_BASE_URLS
        ]
        try:
            pocket_data = _fetch_json_from_candidates(_dedupe_preserve_order(pocket_urls))
        except HttpTransportError as exc:
            self._disable_rest(f"pocket lookup transport failure ({uniprot_id}): {exc}")
            return None

        # pocket_data is a list of 85 dicts with "residue.id"
        if not pocket_data or len(pocket_data) != KLIFS_POCKET_SIZE:
            log.warning("KLIFS pocket for structure %d has %d residues (expected 85)",
                        best_struct_id, len(pocket_data) if pocket_data else 0)
            return None

        result: list[int | None] = []
        for entry in pocket_data:
            rn = _extract_first(entry, ("residue.id", "residue_id", "residue_number"))
            if rn is None:
                result.append(None)
                continue
            try:
                result.append(int(rn))
            except (TypeError, ValueError):
                result.append(None)

        return result if any(x is not None for x in result) else None


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# AlphaFold structure fetcher
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fetch_alphafold_structure(
    uniprot_id: str,
    klifs_indices: list[int | None],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fetch AlphaFold CÎ± coordinates and pLDDT for the KLIFS pocket.

    Returns
    -------
    coords  : (85, 3)  CÎ± Ã… coordinates (zeros for missing)
    plddt   : (85,)    pLDDT scores [0, 100] (0.5*100=50 for missing)
    """
    coords_arr = np.zeros((KLIFS_POCKET_SIZE, 3), dtype=np.float32)
    plddt_arr  = np.full(KLIFS_POCKET_SIZE, 50.0, dtype=np.float32)  # neutral fallback

    # Try to fetch the AlphaFold CIF model
    url = f"{ALPHAFOLD_API_BASE}/prediction/{uniprot_id}"
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200 or not resp.json():
            log.debug("%s: AlphaFold not available - using zero coords fallback.", uniprot_id)
            return torch.tensor(coords_arr), torch.tensor(plddt_arr)

        model_url = resp.json()[0].get("cifUrl") or resp.json()[0].get("pdbUrl")
        if not model_url:
            return torch.tensor(coords_arr), torch.tensor(plddt_arr)

        struct_resp = requests.get(model_url, timeout=60)
        struct_resp.raise_for_status()
    except Exception as exc:
        log.debug("%s: AlphaFold fetch failed (%s) - zero coords fallback.", uniprot_id, exc)
        return torch.tensor(coords_arr), torch.tensor(plddt_arr)

    # Parse CÎ± from CIF / PDB
    residue_data: dict[int, tuple[list[float], float]] = {}   # resnum â†’ (xyz, plddt)

    if model_url.endswith(".cif"):
        residue_data = _parse_alphafold_cif(struct_resp.text)
    else:
        residue_data = _parse_alphafold_pdb(struct_resp.text)

    # Map to KLIFS positions
    for i, res_num in enumerate(klifs_indices):
        if res_num is None or res_num not in residue_data:
            continue
        xyz, b = residue_data[res_num]
        coords_arr[i] = xyz
        plddt_arr[i]  = b

    return torch.tensor(coords_arr), torch.tensor(plddt_arr)


def _parse_alphafold_cif(cif_text: str) -> dict[int, tuple[list[float], float]]:
    """Extract CÎ± xyz and B-factor (= pLDDT) from mmCIF text."""
    result: dict[int, tuple[list[float], float]] = {}
    for line in cif_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        parts = line.split()
        try:
            if parts[3] == "CA":
                seq_id = int(parts[8])
                x, y, z = float(parts[10]), float(parts[11]), float(parts[12])
                bfac    = float(parts[14])
                if seq_id not in result:
                    result[seq_id] = ([x, y, z], bfac)
        except (IndexError, ValueError):
            continue
    return result


def _parse_alphafold_pdb(pdb_text: str) -> dict[int, tuple[list[float], float]]:
    """Extract CÎ± xyz and B-factor (= pLDDT) from PDB text."""
    result: dict[int, tuple[list[float], float]] = {}
    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        name   = line[12:16].strip()
        if name != "CA":
            continue
        try:
            seq_id = int(line[22:26].strip())
            x      = float(line[30:38].strip())
            y      = float(line[38:46].strip())
            z      = float(line[46:54].strip())
            bfac   = float(line[60:66].strip())
            if seq_id not in result:
                result[seq_id] = ([x, y, z], bfac)
        except (ValueError, IndexError):
            continue
    return result


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ProteinFeatureExtractor â€” full feature construction for one UniProt ID
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ProteinFeatureExtractor:
    """
    Build ProteinFeatures for a single retained UniProt ID.

    Only called AFTER the feasibility gate (validate_protein_feasibility).
    """

    def __init__(self) -> None:
        self.embedder     = ESM2Embedder.get_instance()
        self.klifs_mapper = KLIFSPocketMapper()

    def extract(
        self,
        uniprot_id: str,
        target_name: Optional[str] = None,
        pdb_id: Optional[str] = None,
    ) -> Optional[ProteinFeatures]:
        """
        Build and return ProteinFeatures for one UniProt ID.

        Returns None only on fatal unrecoverable failure.
        AlphaFold unavailability is handled gracefully (zero coords).
        """
        # â”€â”€ Sequence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        sequence = fetch_uniprot_sequence(uniprot_id)
        if sequence is None:
            log.error("%s: sequence unavailable - cannot build features.", uniprot_id)
            return None

        # â”€â”€ KLIFS pocket â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        pocket_def = self.klifs_mapper.get_pocket_definition(
            uniprot_id,
            sequence=sequence,
            target_name=target_name,
            pdb_id=pdb_id,
        )
        if pocket_def is None:
            log.error("%s: KLIFS pocket mapping unavailable - dropping target.", uniprot_id)
            return None
        klifs_indices = pocket_def["positions"]
        pocket_confidence = float(pocket_def.get("confidence", 0.0))
        pocket_method = str(pocket_def.get("method", "unknown"))
        pocket_warnings = list(pocket_def.get("warnings", []))
        has_klifs = bool(pocket_def.get("has_klifs", pocket_method == "klifs"))

        pocket_mask = torch.tensor(
            [x is not None for x in klifs_indices], dtype=torch.bool
        )

        # â”€â”€ ESM-2 embedding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        esm_pocket = self.embedder.embed_sequence(uniprot_id, sequence, klifs_indices)
        if esm_pocket is None:
            log.warning(
                "%s: ESM-2 embedding failed â€” using zero pocket embedding.", uniprot_id
            )
            esm_pocket = torch.zeros(KLIFS_POCKET_SIZE, ESM_EMBED_DIM)

        # â”€â”€ AlphaFold coordinates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        coords, plddt = fetch_alphafold_structure(uniprot_id, klifs_indices)
        has_structure  = coords.abs().sum().item() > 0
        confidence     = (plddt / 100.0) * pocket_confidence

        return ProteinFeatures(
            uniprot_id    = uniprot_id,
            esm_pocket    = esm_pocket,
            coords        = coords,
            plddt         = plddt,
            confidence    = confidence,
            pocket_mask   = pocket_mask,
            klifs_indices = klifs_indices,
            full_sequence = sequence,
            has_structure = has_structure,
            has_klifs     = has_klifs,
            pocket_confidence = pocket_confidence,
            pocket_method = pocket_method,
            pocket_warnings = pocket_warnings,
        )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Feature store builder
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_feature_store(
    retained_targets:  pd.DataFrame,
    store_path:        str | Path = "./pipeline_outputs/protein_feature_store.pt",
    workers:           int = 4,
    use_cache:         bool = True,
) -> dict[str, ProteinFeatures]:
    """
    Build or load the protein feature store for all retained UniProt IDs.

    Only retained targets (those that passed feasibility filtering) are
    processed.  No features are generated for dropped targets.

    Returns
    -------
    dict mapping uniprot_id â†’ ProteinFeatures
    """
    store_path = Path(store_path)

    if use_cache and store_path.exists():
        log.info("Loading protein feature store from cache: %s", store_path)
        store = torch.load(store_path, map_location="cpu", weights_only=False)
        log.info("  %d protein features loaded.", len(store))
        return store

    log.info("=" * 65)
    log.info("Building protein feature store for %d retained targets ...", len(retained_targets))
    log.info("=" * 65)

    extractor = ProteinFeatureExtractor()
    store:    dict[str, ProteinFeatures] = {}
    fail_log: list[dict] = []

    # Serial extraction (ESM-2 is GPU-bound; parallelism over GPU doesn't help)
    for _, row in progress_iter(
        retained_targets.iterrows(),
        total=len(retained_targets),
        desc="Protein features",
    ):
        uid = row["uniprot_id"]
        log.info("  Processing %s ...", uid)
        try:
            features = extractor.extract(
                uid,
                target_name=row.get("target", ""),
                pdb_id=row.get("pdb_id", ""),
            )
            if features is not None:
                store[uid] = features
                log.info("    OK %s - pocket residues: %d",
                         uid, features.pocket_mask.sum().item())
            else:
                log.warning("    FAILED %s - feature extraction returned None.", uid)
                fail_log.append({"uniprot_id": uid, "reason": "extraction_returned_none"})
        except Exception as exc:
            log.error("    FAILED %s - unexpected error: %s", uid, exc)
            fail_log.append({"uniprot_id": uid, "reason": str(exc)})

    torch.save(store, store_path)
    log.info("Protein feature store saved -> %s  (%d / %d targets)",
             store_path, len(store), len(retained_targets))
    if store:
        pocket_conf = np.array([float(feat.pocket_confidence) for feat in store.values()], dtype=float)
        n_klifs = sum(1 for feat in store.values() if feat.has_klifs)
        log.info(
            "Protein pocket methods: KLIFS=%d sequence_fallback=%d | confidence mean=%.3f min=%.3f max=%.3f",
            n_klifs,
            len(store) - n_klifs,
            float(pocket_conf.mean()),
            float(pocket_conf.min()),
            float(pocket_conf.max()),
        )

    if fail_log:
        fail_path = store_path.with_name("protein_build_failures.csv")
        pd.DataFrame(fail_log).to_csv(fail_path, index=False)
        log.warning("  %d targets failed feature construction - see %s",
                    len(fail_log), fail_path)

    return store


def load_protein_feature_store(path: str | Path) -> dict[str, ProteinFeatures]:
    """
    Load the protein feature store from disk.

    Supports both the current `protein_feature_store.pt` filename and the older
    `protein_features.pt` alias used by some downstream modules.
    """
    raw_path = Path(path)
    candidates: list[Path] = [raw_path]
    if raw_path.suffix == "":
        candidates.append(raw_path.with_suffix(".pt"))
    if raw_path.name in {"protein_features", "protein_features.pt"}:
        candidates.append(raw_path.with_name("protein_feature_store.pt"))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            log.info("Loading protein feature store: %s", candidate)
            return torch.load(candidate, map_location="cpu", weights_only=False)

    raise FileNotFoundError(
        f"Protein feature store not found. Tried: {[str(p) for p in candidates]}"
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Entry point
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build or validate protein features for retained kinase targets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--retained",  default="./pipeline_outputs/retained_targets.csv",
                        help="Path to retained_targets.csv from Stage 1")
    parser.add_argument("--store",     default="./pipeline_outputs/protein_feature_store.pt",
                        help="Output protein feature store path")
    parser.add_argument("--no_cache",  action="store_true",
                        help="Ignore cache and rebuild from scratch")
    parser.add_argument("--validate_only", action="store_true",
                        help="Run feasibility check only â€” do NOT build ESM-2 features")
    args = parser.parse_args()

    retained_df = pd.read_csv(args.retained)
    log.info("Loaded %d retained targets from %s", len(retained_df), args.retained)

    if args.validate_only:
        klifs = KLIFSPocketMapper()
        for _, row in retained_df.iterrows():
            result = validate_protein_feasibility(row["uniprot_id"], klifs)
            status = "âœ“" if result["protein_status"] == "retained" else "âœ—"
            print(f"{status} {row['uniprot_id']:12s}  "
                  f"seq={result['seq_length']:5d}  "
                  f"klifs={result['klifs_residues']:2d}  "
                  f"af={result['alphafold_ok']}  "
                  f"drop={result['drop_reason']}")
    else:
        store = build_feature_store(
            retained_df,
            store_path = args.store,
            use_cache  = not args.no_cache,
        )
        print(f"\nFeature store: {len(store)} proteins")
        for uid, feat in list(store.items())[:5]:
            print(f"  {uid}: esm_pocket={tuple(feat.esm_pocket.shape)}, "
                  f"has_structure={feat.has_structure}, "
                  f"pocket_residues={feat.pocket_mask.sum().item()}")


