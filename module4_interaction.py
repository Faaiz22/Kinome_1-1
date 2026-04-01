"""
module4_interaction.py
======================
TITLE
Ligand-protein interaction layers for joint modelling.

PURPOSE
This module contains the interaction logic that combines ligand embeddings with
protein pocket embeddings in a reusable, model-agnostic way.

WHAT IT DOES
- Provides concat and cross-attention interaction blocks.
- Supports confidence-aware protein attention.
- Produces pooled interaction representations for prediction heads.

HOW IT WORKS
1. Receive encoded ligand and protein representations.
2. Project them into a shared interaction space.
3. Combine them with concat or cross-attention logic.
4. Return pooled joint embeddings and optional attention maps.

INPUT CONTRACT
- Ligand embeddings with batch-consistent shapes.
- Protein embeddings and masks aligned to the retained pocket definition.

OUTPUT CONTRACT
- Joint interaction embeddings usable by module5 prediction heads.

DEPENDENCIES
- torch, torch.nn

CRITICAL ASSUMPTIONS
- Upstream encoders produce compatible dimensions.
- Protein masks correctly mark padded residues.

FAILURE MODES
- Shape mismatches
- Invalid masks
- Missing interaction inputs

SAFETY CHECKS IMPLEMENTED
- Configuration validation
- Mask-aware attention logic
- Clear interface boundaries for module5

HOW TO RUN
- Imported by module5_models.py during model construction.

HOW IT CONNECTS TO PIPELINE
It is the central fusion layer between ligand and protein encoders in the joint
model families.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Configuration dataclass
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@dataclass
class InteractionConfig:
    """
    Configuration for the CrossAttentionInteraction module.

    Attributes
    ----------
    d_ligand      : input dimension of each ligand atom node embedding
    d_protein     : input dimension of each protein residue embedding (1280 for ESM-2)
    d_attn        : projected dimension for Q, K, V
    n_heads       : number of parallel attention heads
    dropout       : dropout probability on attention weights
    pooling       : 'mean' | 'attention' | 'max'
    use_protein_confidence : multiply attention by pLDDT confidence
    output_dim    : dimension of the final interaction vector
                    (defaults to d_attn if 0)
    """
    d_ligand:               int   = 256
    d_protein:              int   = 1280
    d_attn:                 int   = 256
    n_heads:                int   = 4
    dropout:                float = 0.1
    pooling:                str   = "mean"          # 'mean' | 'attention' | 'max'
    use_protein_confidence: bool  = False
    output_dim:             int   = 0               # 0 â†’ use d_attn

    def __post_init__(self) -> None:
        assert self.d_attn % self.n_heads == 0, (
            f"d_attn ({self.d_attn}) must be divisible by n_heads ({self.n_heads})"
        )
        assert self.pooling in {"mean", "attention", "max"}, (
            f"Unknown pooling: '{self.pooling}'.  Choose from: mean, attention, max"
        )
        if self.output_dim == 0:
            self.output_dim = self.d_attn


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Attention Pooling (learnable; permutation-invariant)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class AttentionPooling(nn.Module):
    """
    Global attention pooling over the atom dimension.

    Learns a scalar importance score for each atom's embedding,
    then produces a weighted sum.

    Input  : (B, N_atoms, d)
    Output : (B, d)
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.Tanh(),
            nn.Linear(d // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : (B, N, d)
        mask : (B, N) bool tensor â€” True for valid atoms, False for padding
               (None â†’ all atoms are valid)

        Returns
        -------
        (B, d)
        """
        scores = self.gate(x).squeeze(-1)   # (B, N)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (B, N)
        # Handle all-masked edge case
        weights = torch.nan_to_num(weights, nan=0.0)
        return (weights.unsqueeze(-1) * x).sum(dim=1)   # (B, d)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Multi-head cross-attention
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head scaled dot-product cross-attention.

    Q sourced from ligand atom embeddings.
    K, V sourced from protein pocket residue embeddings.

    All inputs are linearly projected to d_attn before attention.

    Input shapes (batched)
    ----------------------
    ligand_emb  : (B, N_atoms,   d_ligand)
    protein_emb : (B, 85,        d_protein)
    confidence  : (B, 85)        optional; multiplied into attention logits via learnable gate
    protein_mask: (B, 85)        True = padding (mask), False = valid
    ligand_mask : (B, N_atoms)   True = valid atom, False = padding

    Output shape
    ------------
    (B, N_atoms, d_attn)
    """

    def __init__(self, cfg: InteractionConfig) -> None:
        super().__init__()
        self.cfg      = cfg
        self.n_heads  = cfg.n_heads
        self.d_head   = cfg.d_attn // cfg.n_heads
        self.scale    = math.sqrt(self.d_head)

        # Projections
        self.q_proj   = nn.Linear(cfg.d_ligand,  cfg.d_attn, bias=False)
        self.k_proj   = nn.Linear(cfg.d_protein, cfg.d_attn, bias=False)
        self.v_proj   = nn.Linear(cfg.d_protein, cfg.d_attn, bias=False)
        self.out_proj = nn.Linear(cfg.d_attn,    cfg.d_attn, bias=False)

        self.attn_drop = nn.Dropout(cfg.dropout)

        # Learnable confidence gating (replaces log(confidence))
        if cfg.use_protein_confidence:
            self.conf_proj = nn.Linear(1, 1, bias=True)

    # ------------------------------------------------------------------
    def forward(
        self,
        ligand_emb:   torch.Tensor,
        protein_emb:  torch.Tensor,
        confidence:   Optional[torch.Tensor] = None,
        protein_mask: Optional[torch.Tensor] = None,
        ligand_mask:  Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        ligand_emb  : (B, N, d_ligand)
        protein_emb : (B, P, d_protein)   P = 85 (KLIFS pocket)
        confidence  : (B, P)              pLDDT/100 for each pocket residue
        protein_mask: (B, P)              True = padding, False = valid
        ligand_mask : (B, N)              True = valid atom, False = padded

        Returns
        -------
        out          : (B, N, d_attn)
        attn_weights : (B, n_heads, N, P)  useful for visualisation
        """
        B, N, _ = ligand_emb.shape
        _, P, _ = protein_emb.shape

        # Project
        Q = self.q_proj(ligand_emb)   # (B, N, d_attn)
        K = self.k_proj(protein_emb)  # (B, P, d_attn)
        V = self.v_proj(protein_emb)  # (B, P, d_attn)

        # Apply ligand mask BEFORE attention (set padded query to 0)
        if ligand_mask is not None:
            invalid_lig = ~ligand_mask  # (B, N)
            Q = Q.masked_fill(invalid_lig.unsqueeze(-1), 0.0)

        # Reshape to multi-head: (B, heads, seq, d_head)
        def _reshape(t: torch.Tensor, seq_len: int) -> torch.Tensor:
            return t.view(B, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        Q = _reshape(Q, N)   # (B, H, N, d_head)
        K = _reshape(K, P)   # (B, H, P, d_head)
        V = _reshape(V, P)   # (B, H, P, d_head)

        # Scaled dot-product attention logits
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # attn_logits shape: (B, H, N, P)

        # Apply protein mask BEFORE softmax: (B, P) True = padding
        if protein_mask is not None:
            mask_expanded = protein_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, P)
            attn_logits = attn_logits.masked_fill(mask_expanded, float("-inf"))

        # Learnable confidence gating (if enabled)
        if confidence is not None and self.cfg.use_protein_confidence:
            conf_gate = torch.sigmoid(self.conf_proj(confidence.unsqueeze(-1))).squeeze(-1)
            # conf_gate: (B, P) â†’ (B, 1, 1, P) for broadcasting
            conf_gate = conf_gate.unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits + conf_gate.log().clamp(min=-10.0)

        attn_weights = F.softmax(attn_logits, dim=-1)       # (B, H, N, P)
        # Replace NaN (all-masked rows) with 0
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.attn_drop(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights, V)                  # (B, H, N, d_head)
        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, N, -1)  # (B, N, d_attn)
        out = self.out_proj(out)

        # Mask ligand padding in output
        if ligand_mask is not None:
            out = out * ligand_mask.unsqueeze(-1).float()

        return out, attn_weights


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Main interaction module
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class CrossAttentionInteraction(nn.Module):
    """
    Full ligandâ€“protein interaction module.

    Steps
    -----
    1. Multi-head cross-attention: ligand atoms attend to pocket residues.
    2. Residual connection + layer norm (pre-norm style).
    3. Feed-forward projection (2-layer MLP with GELU).
    4. Permutation-invariant pooling over atom dimension.
    5. Optional linear projection to output_dim.

    Input shapes
    ------------
    ligand_emb  : (B, N_atoms, d_ligand)
    protein_emb : (B, 85,      d_protein)
    confidence  : (B, 85)        optional
    ligand_mask : (B, N_atoms)   True = valid atom

    Output
    ------
    interaction_vec : (B, output_dim)   fixed-size interaction representation
    attn_weights    : (B, n_heads, N_atoms, 85)   for interpretability
    """

    def __init__(self, cfg: InteractionConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # â”€â”€ Input layer norms â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.ligand_norm  = nn.LayerNorm(cfg.d_ligand)
        self.protein_norm = nn.LayerNorm(cfg.d_protein)

        # â”€â”€ Residual projection (ligand_emb â†’ d_attn if dims differ) â”€â”€â”€â”€
        if cfg.d_ligand != cfg.d_attn:
            self.residual_proj = nn.Linear(cfg.d_ligand, cfg.d_attn, bias=False)
        else:
            self.residual_proj = nn.Identity()

        # â”€â”€ Cross-attention â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.cross_attn = MultiHeadCrossAttention(cfg)

        # â”€â”€ Post-attention layer norm â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.post_attn_norm = nn.LayerNorm(cfg.d_attn)

        # â”€â”€ Feed-forward after attention â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        ff_dim = cfg.d_attn * 4
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_attn, ff_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(ff_dim, cfg.d_attn),
            nn.Dropout(cfg.dropout),
        )
        self.ff_norm = nn.LayerNorm(cfg.d_attn)

        # â”€â”€ Pooling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if cfg.pooling == "attention":
            self.pool = AttentionPooling(cfg.d_attn)
        else:
            self.pool = None    # mean / max handled in forward()

        # â”€â”€ Protein summary projection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.protein_summary_proj = nn.Linear(cfg.d_protein, cfg.d_attn, bias=True)

        # â”€â”€ Output projection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Account for protein summary concatenation (d_attn + d_attn â†’ output_dim)
        final_dim = cfg.d_attn * 2  # ligand_pooled (d_attn) + protein_summary (d_attn)
        if cfg.output_dim != final_dim:
            self.out_proj: Optional[nn.Linear] = nn.Linear(
                final_dim, cfg.output_dim, bias=False
            )
        else:
            self.out_proj = None

    # ------------------------------------------------------------------
    def forward(
        self,
        ligand_emb:   torch.Tensor,
        protein_emb:  torch.Tensor,
        confidence:   Optional[torch.Tensor] = None,
        protein_mask: Optional[torch.Tensor] = None,
        ligand_mask:  Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        interaction_vec : (B, output_dim)
        attn_weights    : (B, n_heads, N_atoms, 85)
        """
        # â”€â”€ Pre-norm on inputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        lig  = self.ligand_norm(ligand_emb)
        prot = self.protein_norm(protein_emb)

        # â”€â”€ Cross-attention â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        attended, attn_weights = self.cross_attn(
            lig, prot,
            confidence=confidence,
            protein_mask=protein_mask,
            ligand_mask=ligand_mask,
        )   # (B, N, d_attn)

        # â”€â”€ Residual projection (use pre-registered layer from __init__) â”€
        residual = self.residual_proj(ligand_emb)   # (B, N, d_attn)

        # Post-attention residual + norm
        attended = self.post_attn_norm(attended + residual)

        # â”€â”€ Feed-forward â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        ff_out = self.ff(attended)
        attended = self.ff_norm(ff_out + attended)   # (B, N, d_attn)

        # â”€â”€ Pooling (N_atoms â†’ scalar vector) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.cfg.pooling == "attention":
            pooled = self.pool(attended, mask=ligand_mask)   # (B, d_attn)

        elif self.cfg.pooling == "max":
            if ligand_mask is not None:
                mask_exp = ligand_mask.unsqueeze(-1).float()   # (B, N, 1)
                attended_masked = attended * mask_exp + (1 - mask_exp) * (-1e9)
            else:
                attended_masked = attended
            pooled = attended_masked.max(dim=1).values          # (B, d_attn)

        else:  # "mean" â€” default
            if ligand_mask is not None:
                mask_f = ligand_mask.float().unsqueeze(-1)      # (B, N, 1)
                pooled = (attended * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            else:
                pooled = attended.mean(dim=1)                   # (B, d_attn)

        # â”€â”€ Compute protein summary (masked mean) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        prot_proj = self.protein_summary_proj(protein_emb)  # (B, 85, d_attn)

        if protein_mask is not None:
            valid_mask = (~protein_mask).float().unsqueeze(-1)   # (B, 85, 1)
            protein_summary = (prot_proj * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            protein_summary = prot_proj.mean(dim=1)             # (B, d_attn)

        # â”€â”€ Concatenate ligand pooled + protein summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        combined = torch.cat([pooled, protein_summary], dim=-1)  # (B, 2*d_attn)

        # â”€â”€ Output projection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.out_proj is not None:
            combined = self.out_proj(combined)   # (B, output_dim)

        return combined, attn_weights

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"d_ligand={self.cfg.d_ligand}, d_protein={self.cfg.d_protein}, "
            f"d_attn={self.cfg.d_attn}, n_heads={self.cfg.n_heads}, "
            f"pooling='{self.cfg.pooling}', "
            f"use_confidence={self.cfg.use_protein_confidence}"
        )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Concatenation baseline (ablation)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class ConcatInteraction(nn.Module):
    """
    Simple baseline: pool the ligand embedding independently, then
    concatenate with a pooled protein embedding and protein summary.

    No cross-attention; used as an ablation to measure the contribution
    of the interaction module.

    Input
    -----
    ligand_emb  : (B, N_atoms, d_ligand)
    protein_emb : (B, 85,      d_protein)
    protein_mask: (B, 85)        True = padding, False = valid

    Output
    ------
    (B, d_ligand_proj + d_prot_proj + d_prot_summary)
    """

    def __init__(self, d_ligand: int, d_protein: int, d_proj: int = 256) -> None:
        super().__init__()
        self.lig_proj  = nn.Linear(d_ligand,  d_proj)
        self.prot_proj = nn.Linear(d_protein, d_proj)
        self.prot_summary_proj = nn.Linear(d_protein, d_proj)
        self.out_dim   = d_proj * 3

    def forward(
        self,
        ligand_emb:   torch.Tensor,
        protein_emb:  torch.Tensor,
        confidence:   Optional[torch.Tensor] = None,
        protein_mask: Optional[torch.Tensor] = None,
        ligand_mask:  Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, None]:
        # Pool ligand
        if ligand_mask is not None:
            mask_f = ligand_mask.float().unsqueeze(-1)
            lig_pooled = (ligand_emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            lig_pooled = ligand_emb.mean(dim=1)   # (B, d_ligand)

        # Pool protein (main)
        if protein_mask is not None:
            valid_mask = (~protein_mask).float().unsqueeze(-1)  # (B, 85, 1)
            prot_pooled = (protein_emb * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            prot_pooled = protein_emb.mean(dim=1)              # (B, d_protein)

        # Protein summary (identical to main pool for now, but separate projection)
        prot_summary = self.prot_summary_proj(prot_pooled)

        lig_proj  = F.gelu(self.lig_proj(lig_pooled))    # (B, d_proj)
        prot_proj = F.gelu(self.prot_proj(prot_pooled))  # (B, d_proj)
        prot_summ = F.gelu(prot_summary)                 # (B, d_proj)

        return torch.cat([lig_proj, prot_proj, prot_summ], dim=-1), None


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Factory function
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_interaction_module(
    mode: str,
    d_ligand: int,
    d_protein: int,
    d_attn: int = 256,
    n_heads: int = 4,
    dropout: float = 0.1,
    pooling: str = "mean",
    use_protein_confidence: bool = False,
    output_dim: int = 0,
) -> nn.Module:
    """
    Factory function to instantiate the appropriate interaction module.

    Parameters
    ----------
    mode : 'cross_attention' | 'concat'
    All others: forwarded to InteractionConfig or ConcatInteraction.

    Returns
    -------
    nn.Module (CrossAttentionInteraction or ConcatInteraction)
    """
    if mode == "cross_attention":
        cfg = InteractionConfig(
            d_ligand               = d_ligand,
            d_protein              = d_protein,
            d_attn                 = d_attn,
            n_heads                = n_heads,
            dropout                = dropout,
            pooling                = pooling,
            use_protein_confidence = use_protein_confidence,
            output_dim             = output_dim,
        )
        return CrossAttentionInteraction(cfg)

    elif mode == "concat":
        d_proj = d_attn if d_attn else 256
        return ConcatInteraction(d_ligand=d_ligand, d_protein=d_protein, d_proj=d_proj)

    else:
        raise ValueError(
            f"Unknown interaction mode: '{mode}'. "
            f"Choose from: 'cross_attention', 'concat'."
        )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Quick self-test (no training, no data required)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("module4_test")

    B, N_atoms, d_lig, d_prot = 4, 32, 256, 1280

    ligand_emb  = torch.randn(B, N_atoms, d_lig)
    protein_emb = torch.randn(B, 85, d_prot)
    confidence  = torch.rand(B, 85)
    protein_mask = torch.zeros(B, 85, dtype=torch.bool)
    protein_mask[:, -5:] = True  # mask last 5 residues (padding)
    ligand_mask = torch.ones(B, N_atoms, dtype=torch.bool)
    ligand_mask[:, -5:] = False   # simulate 5 padded atoms

    # â”€â”€ Cross-attention (mean pool) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cfg = InteractionConfig(
        d_ligand=d_lig, d_protein=d_prot, d_attn=256,
        n_heads=4, pooling="mean", use_protein_confidence=True,
    )
    model_ca = CrossAttentionInteraction(cfg)
    out, attn = model_ca(ligand_emb, protein_emb, confidence, protein_mask, ligand_mask)
    # Output now includes protein summary: d_attn * 2 (before projection)
    final_dim = cfg.output_dim if cfg.output_dim else cfg.d_attn * 2
    assert out.shape  == (B, final_dim), f"Expected (4, {final_dim}), got {out.shape}"
    assert attn.shape == (B, 4, N_atoms, 85), f"Attn shape error: {attn.shape}"
    assert not torch.isnan(out).any(),   "NaN in cross-attention output"
    log.info("CrossAttention (mean)  âœ“  out=%s  attn=%s", out.shape, attn.shape)

    # â”€â”€ Cross-attention (attention pool) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cfg2 = InteractionConfig(
        d_ligand=d_lig, d_protein=d_prot, d_attn=256,
        n_heads=4, pooling="attention",
    )
    model_ap = CrossAttentionInteraction(cfg2)
    out2, _ = model_ap(ligand_emb, protein_emb, confidence, protein_mask, ligand_mask)
    final_dim2 = cfg2.output_dim if cfg2.output_dim else cfg2.d_attn * 2
    assert out2.shape == (B, final_dim2), f"Expected (4, {final_dim2}), got {out2.shape}"
    log.info("CrossAttention (attn pool)  âœ“  out=%s", out2.shape)

    # â”€â”€ Concatenation baseline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    model_cat = ConcatInteraction(d_ligand=d_lig, d_protein=d_prot, d_proj=256)
    out3, _ = model_cat(ligand_emb, protein_emb, confidence, protein_mask, ligand_mask)
    # Output now includes protein summary: d_proj * 3
    assert out3.shape == (B, 768), f"Expected (4, 768), got {out3.shape}"
    log.info("ConcatInteraction           âœ“  out=%s", out3.shape)

    # â”€â”€ Factory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    m = build_interaction_module(
        "cross_attention", d_lig, d_prot,
        d_attn=256, n_heads=4, pooling="max", output_dim=128,
    )
    out4, _ = m(ligand_emb, protein_emb, confidence, protein_mask, ligand_mask)
    assert out4.shape == (B, 128), f"Expected (4, 128), got {out4.shape}"
    log.info("Factory (max pool, out=128) âœ“  out=%s", out4.shape)

    print("\nAll module4 self-tests passed âœ“")


