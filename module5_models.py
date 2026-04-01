"""
module5_models.py
=================
TITLE
Unified model zoo for kinase-ligand prediction.

PURPOSE
This module defines the trainable neural architectures used across regression,
classification, uncertainty estimation, and downstream inference.

WHAT IT DOES
- Defines model configurations and architecture variants.
- Builds ligand encoders, protein encoders, interaction layers, and prediction
  heads.
- Exposes one `BaseModel` interface for all supported configs.

HOW IT WORKS
1. Resolve a named model configuration.
2. Build ligand and protein encoders.
3. Fuse encodings with interaction logic.
4. Produce predictions and, when configured, uncertainty outputs.

INPUT CONTRACT
- Batch dictionaries matching module6 collate output.
- Valid config IDs from the registered config set.

OUTPUT CONTRACT
- Regression outputs: `(mu, log_var)`.
- Classification outputs: logits.
- Parameter metadata via model utility helpers.

DEPENDENCIES
- torch, torch_geometric
- module4_interaction.py

CRITICAL ASSUMPTIONS
- Feature tensor dimensions match module2 and module3 contracts.
- Config IDs are treated as public API across the pipeline.

FAILURE MODES
- Unknown config IDs
- Shape mismatches between encoders and heads
- Missing required batch fields for a selected model family

SAFETY CHECKS IMPLEMENTED
- Config validation
- Explicit forward-path checks for required tensors
- Shared base model interface to reduce branching errors

HOW TO RUN
- Imported by module6_training.py and module10_streamlit.py.

HOW IT CONNECTS TO PIPELINE
This is the core trainable modelling layer used by training, evaluation,
uncertainty estimation, experiments, and inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    GINEConv,
    GlobalAttention,
    global_add_pool,
    global_mean_pool,
)
from torch_geometric.utils import to_dense_batch

from module4_interaction import (
    CrossAttentionInteraction,
    ConcatInteraction,
    InteractionConfig,
    build_interaction_module,
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Feature dimension constants (must match module2 / module3)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ATOM_FEAT_DIM:     int = 43
BOND_FEAT_DIM:     int = 12
PHYSCHEM_DIM:      int = 22
ESM_EMBED_DIM:     int = 1280
KLIFS_POCKET_SIZE: int = 85


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Model configuration dataclass
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@dataclass
class ModelConfig:
    """
    Single dataclass controlling all model behaviour.

    Ligand encoder
    ---------------
    ligand_encoder     : 'gnn' | 'mlp'
    gnn_type           : 'gine' | 'gat'
    gnn_layers         : GNN message-passing depth (kept â‰¤ 6 for trainability)
    gnn_hidden         : GNN hidden dimension (â‰¤ 256 recommended)
    gnn_dropout        : dropout in GNN

    Protein encoder
    ---------------
    protein_encoder    : 'mlp' | 'transformer' | 'none'
    prot_hidden        : protein encoder output dim
    prot_transformer_layers : transformer encoder depth (â‰¤ 3 recommended)

    Interaction
    -----------
    interaction        : 'cross_attention' | 'concat' | 'none'
    interaction_d_attn : cross-attention projection dim
    interaction_heads  : number of attention heads
    use_physchem       : append physicochemical features to head input
    use_protein_confidence : weight attention by pLDDT
    interaction_pooling: 'mean' | 'attention'

    Output head
    -----------
    head_hidden        : MLP head hidden dim
    head_layers        : MLP head depth
    head_dropout       : dropout in head
    use_uncertainty    : if True, output (mu, log_var); else (mu, 0)
    task_type          : 'regression' | 'classification'

    Meta
    ----
    config_id          : string identifier used in logging / results
    d_model            : shared embedding dimension
    dropout            : global dropout fallback
    """
    # â”€â”€ Ligand encoder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ligand_encoder:          str   = "gnn"
    gnn_type:                str   = "gine"
    gnn_layers:              int   = 4
    gnn_hidden:              int   = 256
    gnn_dropout:             float = 0.1

    # â”€â”€ Protein encoder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    protein_encoder:         str   = "mlp"
    prot_hidden:             int   = 256
    prot_transformer_layers: int   = 2

    # â”€â”€ Interaction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    interaction:             str   = "cross_attention"
    interaction_d_attn:      int   = 256
    interaction_heads:       int   = 4
    use_physchem:            bool  = True
    use_protein_confidence:  bool  = False
    interaction_pooling:     str   = "mean"

    # â”€â”€ Output head â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    head_hidden:             int   = 256
    head_layers:             int   = 3
    head_dropout:            float = 0.1
    use_uncertainty:         bool  = True
    use_generative:          bool  = False
    task_type:               str   = "regression"

    # â”€â”€ Meta â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    config_id:               str   = "full_model"
    d_model:                 int   = 256
    dropout:                 float = 0.1
    family:                  str   = "regression"
    label_scheme:            str   = "pIC50_continuous"


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5 pre-defined configurations (compact, 24-hour trainable)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_CONFIGS: dict[str, ModelConfig] = {

    # â”€â”€ 1. Ligand-only baseline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "ligand_only": ModelConfig(
        config_id        = "ligand_only",
        ligand_encoder   = "gnn",
        gnn_type         = "gine",
        gnn_layers       = 3,
        gnn_hidden       = 128,
        protein_encoder  = "none",
        interaction      = "none",
        use_physchem     = False,
        use_uncertainty  = False,
        head_hidden      = 128,
        head_layers      = 2,
        d_model          = 128,
    ),

    # â”€â”€ 2. Protein-only baseline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "protein_only": ModelConfig(
        config_id        = "protein_only",
        ligand_encoder   = "mlp",          # physchem only as ligand rep
        gnn_type         = "gine",
        protein_encoder  = "mlp",
        prot_hidden      = 128,
        interaction      = "none",
        use_physchem     = True,
        use_uncertainty  = False,
        head_hidden      = 128,
        head_layers      = 2,
        d_model          = 128,
    ),

    # â”€â”€ 3. Ligand + protein, concat interaction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "ligand_plus_protein": ModelConfig(
        config_id        = "ligand_plus_protein",
        ligand_encoder   = "gnn",
        gnn_type         = "gine",
        gnn_layers       = 4,
        gnn_hidden       = 192,
        protein_encoder  = "mlp",
        prot_hidden      = 192,
        interaction      = "concat",
        use_physchem     = False,
        use_uncertainty  = True,
        head_hidden      = 192,
        head_layers      = 3,
        d_model          = 192,
    ),

    # â”€â”€ 4. Cross-attention model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "cross_attention": ModelConfig(
        config_id                = "cross_attention",
        ligand_encoder           = "gnn",
        gnn_type                 = "gine",
        gnn_layers               = 4,
        gnn_hidden               = 256,
        protein_encoder          = "transformer",
        prot_hidden              = 256,
        prot_transformer_layers  = 2,
        interaction              = "cross_attention",
        interaction_d_attn       = 256,
        interaction_heads        = 4,
        interaction_pooling      = "mean",
        use_physchem             = False,
        use_protein_confidence   = False,
        use_uncertainty          = True,
        head_hidden              = 256,
        head_layers              = 3,
        d_model                  = 256,
    ),

    # â”€â”€ 5. Full model (recommended for dissertation) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "full_model": ModelConfig(
        config_id                = "full_model",
        ligand_encoder           = "gnn",
        gnn_type                 = "gine",
        gnn_layers               = 5,
        gnn_hidden               = 256,
        protein_encoder          = "transformer",
        prot_hidden              = 256,
        prot_transformer_layers  = 3,
        interaction              = "cross_attention",
        interaction_d_attn       = 256,
        interaction_heads        = 4,
        interaction_pooling      = "attention",
        use_physchem             = True,
        use_protein_confidence   = True,
        use_uncertainty          = True,
        head_hidden              = 256,
        head_layers              = 3,
        head_dropout             = 0.15,
        d_model                  = 256,
    ),

    # â”€â”€ Classification configurations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    # â”€â”€ 6. Ligand-only classification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "ligand_only_cls": ModelConfig(
        config_id        = "ligand_only_cls",
        ligand_encoder   = "gnn",
        gnn_type         = "gine",
        gnn_layers       = 3,
        gnn_hidden       = 128,
        protein_encoder  = "none",
        interaction      = "none",
        use_physchem     = False,
        use_uncertainty  = False,
        task_type        = "classification",
        family           = "classification",
        label_scheme     = "binary_activity",
        head_hidden      = 128,
        head_layers      = 2,
        d_model          = 128,
    ),

    # â”€â”€ 7. Protein-only classification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "protein_only_cls": ModelConfig(
        config_id        = "protein_only_cls",
        ligand_encoder   = "mlp",          # physchem only as ligand rep
        gnn_type         = "gine",
        protein_encoder  = "mlp",
        prot_hidden      = 128,
        interaction      = "none",
        use_physchem     = True,
        use_uncertainty  = False,
        task_type        = "classification",
        family           = "classification",
        label_scheme     = "binary_activity",
        head_hidden      = 128,
        head_layers      = 2,
        d_model          = 128,
    ),

    # â”€â”€ 8. Ligand + protein, concat interaction classification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "ligand_plus_protein_cls": ModelConfig(
        config_id        = "ligand_plus_protein_cls",
        ligand_encoder   = "gnn",
        gnn_type         = "gine",
        gnn_layers       = 4,
        gnn_hidden       = 192,
        protein_encoder  = "mlp",
        prot_hidden      = 192,
        interaction      = "concat",
        use_physchem     = False,
        use_uncertainty  = False,
        task_type        = "classification",
        family           = "classification",
        label_scheme     = "binary_activity",
        head_hidden      = 192,
        head_layers      = 3,
        d_model          = 192,
    ),

    # â”€â”€ 9. Cross-attention classification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "cross_attention_cls": ModelConfig(
        config_id                = "cross_attention_cls",
        ligand_encoder           = "gnn",
        gnn_type                 = "gine",
        gnn_layers               = 4,
        gnn_hidden               = 256,
        protein_encoder          = "transformer",
        prot_hidden              = 256,
        prot_transformer_layers  = 2,
        interaction              = "cross_attention",
        interaction_d_attn       = 256,
        interaction_heads        = 4,
        interaction_pooling      = "mean",
        use_physchem             = False,
        use_protein_confidence   = False,
        use_uncertainty          = False,
        task_type                = "classification",
        family                   = "classification",
        label_scheme             = "binary_activity",
        head_hidden              = 256,
        head_layers              = 3,
        d_model                  = 256,
    ),

    # â”€â”€ 10. Full model classification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "full_model_cls": ModelConfig(
        config_id                = "full_model_cls",
        ligand_encoder           = "gnn",
        gnn_type                 = "gine",
        gnn_layers               = 5,
        gnn_hidden               = 256,
        protein_encoder          = "transformer",
        prot_hidden              = 256,
        prot_transformer_layers  = 3,
        interaction              = "cross_attention",
        interaction_d_attn       = 256,
        interaction_heads        = 4,
        interaction_pooling      = "attention",
        use_physchem             = True,
        use_protein_confidence   = True,
        use_uncertainty          = False,
        task_type                = "classification",
        family                   = "classification",
        label_scheme             = "binary_activity",
        head_hidden              = 256,
        head_layers              = 3,
        head_dropout             = 0.15,
        d_model                  = 256,
    ),
}

ALL_CONFIG_IDS: list[str] = list(_CONFIGS.keys())


def get_model_config(config_id: str) -> ModelConfig:
    """Return the ModelConfig for the given config_id."""
    if config_id not in _CONFIGS:
        raise ValueError(
            f"Unknown config_id: '{config_id}'. "
            f"Valid options: {sorted(_CONFIGS.keys())}"
        )
    return _CONFIGS[config_id]


def list_configs() -> list[str]:
    """Return all available configuration names."""
    return sorted(_CONFIGS.keys())


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Utility layers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _make_mlp(
    in_dim:   int,
    hidden:   int,
    out_dim:  int,
    layers:   int = 2,
    dropout:  float = 0.1,
    activate_last: bool = False,
) -> nn.Sequential:
    """Build a fully-connected MLP with BatchNorm + GELU activations."""
    assert layers >= 1
    dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
    mods: list[nn.Module] = []
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = (i == len(dims) - 2)
        if not is_last or activate_last:
            mods.append(nn.LayerNorm(dims[i + 1]))
            mods.append(nn.GELU())
            if dropout > 0:
                mods.append(nn.Dropout(dropout))
    return nn.Sequential(*mods)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Ligand encoders
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class GINELigandEncoder(nn.Module):
    """
    Graph Isomorphism Network with Edge features (GINE).
    Encodes atomic graphs into a single graph-level embedding.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(ATOM_FEAT_DIM, cfg.gnn_hidden)
        self.edge_proj  = nn.Linear(BOND_FEAT_DIM, cfg.gnn_hidden)
        self.convs: nn.ModuleList = nn.ModuleList()
        self.norms: nn.ModuleList = nn.ModuleList()

        for _ in range(cfg.gnn_layers):
            mlp = _make_mlp(
                cfg.gnn_hidden, cfg.gnn_hidden, cfg.gnn_hidden,
                layers=2, dropout=0.0,
            )
            self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=cfg.gnn_hidden))
            self.norms.append(nn.LayerNorm(cfg.gnn_hidden))

        self.dropout  = nn.Dropout(cfg.gnn_dropout)
        self.pool_mlp = nn.Linear(cfg.gnn_hidden, cfg.gnn_hidden)   # attention pooling gate
        self.pool     = GlobalAttention(gate_nn=self.pool_mlp)
        self.out_proj = _make_mlp(cfg.gnn_hidden, cfg.gnn_hidden, cfg.d_model,
                                  layers=2, dropout=cfg.gnn_dropout)
        self.out_dim  = cfg.d_model

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = self.input_proj(x)
        e = self.edge_proj(edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.gelu(conv(h, edge_index, e)) + h)
            h = self.dropout(h)
        # Attention pooling â†’ (B, d_model)
        g = self.pool(h, batch)
        return self.out_proj(g)


class GATLigandEncoder(nn.Module):
    """GATv2 ligand encoder with attention pooling."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        heads = cfg.interaction_heads
        assert cfg.gnn_hidden % heads == 0, (
            f"gnn_hidden ({cfg.gnn_hidden}) must be divisible by heads ({heads})"
        )
        head_dim = cfg.gnn_hidden // heads
        self.input_proj = nn.Linear(ATOM_FEAT_DIM, cfg.gnn_hidden)
        self.convs = nn.ModuleList([
            GATv2Conv(
                cfg.gnn_hidden, head_dim,
                heads=heads, dropout=cfg.gnn_dropout,
                edge_dim=BOND_FEAT_DIM, concat=True,
            )
            for _ in range(cfg.gnn_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(cfg.gnn_hidden) for _ in range(cfg.gnn_layers)
        ])
        self.pool    = GlobalAttention(gate_nn=nn.Linear(cfg.gnn_hidden, 1))
        self.out_proj = _make_mlp(cfg.gnn_hidden, cfg.gnn_hidden, cfg.d_model,
                                  layers=2, dropout=cfg.gnn_dropout)
        self.out_dim  = cfg.d_model

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.gelu(conv(h, edge_index, edge_attr)) + h)
        return self.out_proj(self.pool(h, batch))


class MLPLigandEncoder(nn.Module):
    """
    Fallback ligand encoder using only physicochemical descriptors.
    Used when ligand_encoder='mlp' (e.g. protein_only baseline).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.net     = _make_mlp(PHYSCHEM_DIM, cfg.d_model, cfg.d_model,
                                 layers=2, dropout=cfg.dropout, activate_last=True)
        self.out_dim = cfg.d_model

    def forward(self, physchem: torch.Tensor, **_) -> torch.Tensor:
        return self.net(physchem)


def build_ligand_encoder(cfg: ModelConfig) -> nn.Module:
    if cfg.ligand_encoder == "gnn":
        if cfg.gnn_type == "gat":
            return GATLigandEncoder(cfg)
        return GINELigandEncoder(cfg)
    elif cfg.ligand_encoder == "mlp":
        return MLPLigandEncoder(cfg)
    else:
        raise ValueError(f"Unknown ligand_encoder: '{cfg.ligand_encoder}'")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Protein encoders
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class MLPProteinEncoder(nn.Module):
    """
    Simple MLP over the flattened 85Ã—1280 ESM-2 pocket embedding.

    Projects to (B, prot_hidden) via mean pooling then MLP.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.proj    = _make_mlp(
            ESM_EMBED_DIM, cfg.prot_hidden, cfg.prot_hidden,
            layers=2, dropout=cfg.dropout, activate_last=True,
        )
        self.pool_proj = _make_mlp(
            cfg.prot_hidden, cfg.prot_hidden, cfg.prot_hidden,
            layers=1, dropout=0.0, activate_last=True,
        )
        self.out_dim = cfg.prot_hidden

    def forward(
        self,
        esm_pocket: torch.Tensor,            # (B, 85, 1280)
        pocket_mask: Optional[torch.Tensor] = None,  # (B, 85)
        confidence:  Optional[torch.Tensor] = None,  # (B, 85) ignored for MLP
    ) -> torch.Tensor:
        # Project each residue embedding
        h = self.proj(esm_pocket)            # (B, 85, prot_hidden)
        # Mean pool over residues
        if pocket_mask is not None:
            mask = pocket_mask.unsqueeze(-1).float()
            h    = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        else:
            h = h.mean(1)
        return self.pool_proj(h)


class TransformerProteinEncoder(nn.Module):
    """
    Lightweight Transformer over the 85-residue ESM-2 pocket embeddings.

    Returns both:
        sequence output (B, 85, prot_hidden) for cross-attention
        pooled output   (B, prot_hidden) for concat-style interaction
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(ESM_EMBED_DIM, cfg.prot_hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model    = cfg.prot_hidden,
            nhead      = min(cfg.interaction_heads, cfg.prot_hidden // 64),
            dim_feedforward = cfg.prot_hidden * 2,
            dropout    = cfg.dropout,
            batch_first= True,
            activation = "gelu",
            norm_first = True,   # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=cfg.prot_transformer_layers,
        )
        self.out_dim = cfg.prot_hidden

    def forward(
        self,
        esm_pocket: torch.Tensor,            # (B, 85, 1280)
        pocket_mask: Optional[torch.Tensor] = None,
        confidence:  Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (seq_out, pooled_out).
            seq_out    : (B, 85, prot_hidden)  for cross-attention
            pooled_out : (B, prot_hidden)       for concat
        """
        h = self.input_proj(esm_pocket)  # (B, 85, prot_hidden)

        src_key_padding_mask = None
        if pocket_mask is not None:
            # True â†’ ignore (TransformerEncoder convention)
            src_key_padding_mask = ~pocket_mask.bool()

        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        # pLDDT-weighted pooling
        if confidence is not None:
            w = confidence.unsqueeze(-1)  # (B, 85, 1)
            if pocket_mask is not None:
                w = w * pocket_mask.unsqueeze(-1).float()
            pooled = (h * w).sum(1) / w.sum(1).clamp(min=1e-6)
        elif pocket_mask is not None:
            mask   = pocket_mask.unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        else:
            pooled = h.mean(1)

        return h, pooled


def build_protein_encoder(cfg: ModelConfig) -> Optional[nn.Module]:
    if cfg.protein_encoder == "none":
        return None
    elif cfg.protein_encoder == "mlp":
        return MLPProteinEncoder(cfg)
    elif cfg.protein_encoder == "transformer":
        return TransformerProteinEncoder(cfg)
    else:
        raise ValueError(f"Unknown protein_encoder: '{cfg.protein_encoder}'")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Output head
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class PredictionHead(nn.Module):
    """
    Final MLP prediction head.

    Outputs:
        mu      : (B, 1)  predicted pIC50 or logit
        log_var : (B, 1)  log predicted aleatoric variance (or zeros)
    """

    def __init__(self, in_dim: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.task_type      = cfg.task_type
        self.use_uncertainty = cfg.use_uncertainty

        self.net = _make_mlp(
            in_dim, cfg.head_hidden, cfg.head_hidden,
            layers     = max(cfg.head_layers - 1, 1),
            dropout    = cfg.head_dropout,
            activate_last = True,
        )
        self.mu_head = nn.Linear(cfg.head_hidden, 1)
        if cfg.use_uncertainty:
            self.logvar_head = nn.Linear(cfg.head_hidden, 1)
        else:
            self.logvar_head = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h   = self.net(x)
        mu  = self.mu_head(h)
        if self.logvar_head is not None:
            log_var = self.logvar_head(h)
        else:
            log_var = torch.zeros_like(mu)
        return mu, log_var


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Physicochemical feature projector
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class PhyschemProjector(nn.Module):
    """Project 22-dim physchem descriptors to d_model."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.net = _make_mlp(
            PHYSCHEM_DIM, cfg.d_model, cfg.d_model,
            layers=2, dropout=cfg.dropout, activate_last=True,
        )

    def forward(self, physchem: torch.Tensor) -> torch.Tensor:
        return self.net(physchem)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# BaseModel â€” unified model class for all 5 configurations
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class BaseModel(nn.Module):
    """
    Unified model supporting all 5 configurations.

    Behaviour is controlled entirely by ModelConfig â€” no subclassing required.

    Forward signature
    -----------------
    Required:
        x          : (N_atoms, ATOM_FEAT_DIM)  atom features
        edge_index : (2, N_edges)
        edge_attr  : (N_edges, BOND_FEAT_DIM)
        batch      : (N_atoms,) batch assignment

    Optional:
        esm_pocket  : (B, 85, ESM_EMBED_DIM)
        pocket_mask : (B, 85)   boolean
        confidence  : (B, 85)   pLDDT/100
        physchem    : (B, PHYSCHEM_DIM)

    Returns
    -------
    (mu, log_var) â€” both (B, 1)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # â”€â”€ Ligand encoder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.ligand_encoder = build_ligand_encoder(cfg)
        lig_dim = self.ligand_encoder.out_dim

        # â”€â”€ Protein encoder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.protein_encoder = build_protein_encoder(cfg)
        prot_dim = cfg.prot_hidden if self.protein_encoder is not None else 0

        # â”€â”€ Interaction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if cfg.interaction == "none" or self.protein_encoder is None:
            self.interaction_module = None
            interact_out_dim = lig_dim
        elif cfg.interaction == "concat":
            self.interaction_module = ConcatInteraction(
                d_ligand=lig_dim, d_protein=prot_dim, d_proj=cfg.d_model,
            )
            interact_out_dim = self.interaction_module.out_dim
        elif cfg.interaction == "cross_attention":
            int_cfg = InteractionConfig(
                d_ligand=lig_dim,
                d_protein=prot_dim,
                d_attn=cfg.interaction_d_attn,
                n_heads=cfg.interaction_heads,
                pooling=cfg.interaction_pooling,
                dropout=cfg.dropout,
                use_protein_confidence=cfg.use_protein_confidence,
                output_dim=cfg.interaction_d_attn,
            )
            self.interaction_module = CrossAttentionInteraction(int_cfg)
            interact_out_dim = int_cfg.output_dim
        else:
            raise ValueError(f"Unknown interaction: '{cfg.interaction}'")

        # â”€â”€ Physchem projector â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if cfg.use_physchem:
            self.physchem_proj = PhyschemProjector(cfg)
            head_in_dim = interact_out_dim + cfg.d_model
        else:
            self.physchem_proj = None
            head_in_dim = interact_out_dim

        # â”€â”€ Output head â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.head = PredictionHead(head_in_dim, cfg)

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def forward(
        self,
        x:           torch.Tensor | dict,
        edge_index:  Optional[torch.Tensor] = None,
        edge_attr:   Optional[torch.Tensor] = None,
        batch:       Optional[torch.Tensor] = None,
        esm_pocket:  Optional[torch.Tensor] = None,
        pocket_mask: Optional[torch.Tensor] = None,
        confidence:  Optional[torch.Tensor] = None,
        physchem:    Optional[torch.Tensor] = None,
        protein_mask: Optional[torch.Tensor] = None,
        return_cvae: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict]:

        # â”€â”€ Ligand encoding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if isinstance(x, dict):
            batch_dict = x
            x = batch_dict["x"]
            edge_index = batch_dict["edge_index"]
            edge_attr = batch_dict["edge_attr"]
            batch = batch_dict["batch"]
            esm_pocket = batch_dict.get("esm_pocket", esm_pocket)
            pocket_mask = batch_dict.get("pocket_mask", pocket_mask)
            confidence = batch_dict.get("confidence", confidence)
            physchem = batch_dict.get("physchem", physchem)
            protein_mask = batch_dict.get("protein_mask", protein_mask)

        if edge_index is None or edge_attr is None or batch is None:
            raise ValueError("BaseModel.forward requires x/edge_index/edge_attr/batch tensors.")

        valid_pocket_mask = pocket_mask.bool() if pocket_mask is not None else None
        protein_padding_mask = protein_mask.bool() if protein_mask is not None else None
        if valid_pocket_mask is None and protein_padding_mask is not None:
            valid_pocket_mask = ~protein_padding_mask
        if protein_padding_mask is None and valid_pocket_mask is not None:
            protein_padding_mask = ~valid_pocket_mask

        if self.cfg.ligand_encoder == "mlp":
            if physchem is None:
                raise ValueError("MLPLigandEncoder requires physchem tensor.")
            lig_emb = self.ligand_encoder(physchem=physchem)
        else:
            lig_emb = self.ligand_encoder(x, edge_index, edge_attr, batch)

        # â”€â”€ Protein encoding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.protein_encoder is None:
            prot_seq = None
            prot_pool = None
        elif isinstance(self.protein_encoder, TransformerProteinEncoder):
            conf = confidence if self.cfg.use_protein_confidence else None
            prot_seq, prot_pool = self.protein_encoder(esm_pocket, valid_pocket_mask, conf)
        else:
            # MLPProteinEncoder returns pooled only
            conf = confidence if self.cfg.use_protein_confidence else None
            prot_pool = self.protein_encoder(esm_pocket, valid_pocket_mask, conf)
            prot_seq  = None

        # â”€â”€ Interaction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.interaction_module is None:
            combined = lig_emb
        elif isinstance(self.interaction_module, CrossAttentionInteraction):
            # Cross-attention needs sequence output of protein
            if prot_seq is None:
                prot_seq = prot_pool.unsqueeze(1) if prot_pool is not None else None
            attn_protein_mask = (
                protein_padding_mask
                if protein_padding_mask is not None
                and prot_seq is not None
                and protein_padding_mask.shape[1] == prot_seq.shape[1]
                else None
            )
            if lig_emb.dim() == 2:
                lig_nodes = lig_emb.unsqueeze(1)
                node_mask = torch.ones(
                    lig_nodes.shape[:2], dtype=torch.bool, device=lig_nodes.device
                )
            else:
                lig_nodes, node_mask = to_dense_batch(lig_emb, batch)
            combined, _ = self.interaction_module(
                lig_nodes,
                prot_seq,
                confidence=conf,
                protein_mask=attn_protein_mask,
                ligand_mask=node_mask,
            )
        else:
            ligand_tokens = lig_emb.unsqueeze(1) if lig_emb.dim() == 2 else lig_emb
            protein_tokens = prot_seq
            if protein_tokens is None and prot_pool is not None:
                protein_tokens = prot_pool.unsqueeze(1)
            concat_protein_mask = (
                protein_padding_mask
                if protein_padding_mask is not None
                and protein_tokens is not None
                and protein_padding_mask.shape[1] == protein_tokens.shape[1]
                else None
            )
            combined, _ = self.interaction_module(
                ligand_tokens,
                protein_tokens,
                confidence=conf,
                protein_mask=concat_protein_mask,
                ligand_mask=None if ligand_tokens.dim() == 2 else torch.ones(
                    ligand_tokens.shape[:2], dtype=torch.bool, device=ligand_tokens.device
                ),
            )

        # â”€â”€ Physchem append â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.physchem_proj is not None and physchem is not None:
            pc_emb   = self.physchem_proj(physchem)
            combined = torch.cat([combined, pc_emb], dim=-1)

        mu, log_var = self.head(combined)
        if self.cfg.task_type == "classification":
            logits = mu.squeeze(-1)
            if return_cvae:
                return logits, torch.zeros_like(logits), {}
            return logits
        if return_cvae:
            return mu, log_var, {}
        return mu, log_var

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self) -> str:
        total = self.count_parameters()
        lines = [
            f"Model config : {self.cfg.config_id}",
            f"Parameters   : {total:,}",
        ]
        for name, module in self.named_children():
            n = sum(p.numel() for p in module.parameters() if p.requires_grad)
            lines.append(f"  {name:<30} : {n:>10,}")
        return "\n".join(lines)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Factory function
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_model(config_id: str) -> BaseModel:
    """
    Instantiate a BaseModel from a config_id string.

    Usage
    -----
        model = build_model("full_model")
        model = build_model("cross_attention")
        model = build_model("ligand_only")
    """
    cfg = get_model_config(config_id)
    return BaseModel(cfg)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Uncertainty-aware loss functions
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def gaussian_nll_loss(
    mu:      torch.Tensor,
    log_var: torch.Tensor,
    target:  torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood loss for heteroscedastic regression.

    L = 0.5 * exp(-log_var) * (mu - target)^2 + 0.5 * log_var
    """
    return 0.5 * (torch.exp(-log_var) * (mu - target) ** 2 + log_var).mean()


def regression_loss(
    mu:      torch.Tensor,
    log_var: torch.Tensor,
    target:  torch.Tensor,
    use_uncertainty: bool = True,
) -> torch.Tensor:
    """Dispatch to NLL or MSE based on use_uncertainty flag."""
    if use_uncertainty:
        return gaussian_nll_loss(mu, log_var, target)
    return F.mse_loss(mu.squeeze(-1), target)


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy for activity classification."""
    return F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Quick self-test
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    import torch

    print("=== Module5 Model Inventory ===")
    for cid in list_configs():
        model = build_model(cid)
        print(f"\n{model.parameter_summary()}")

    print("\nâœ“ All 10 model configurations instantiated successfully.")


