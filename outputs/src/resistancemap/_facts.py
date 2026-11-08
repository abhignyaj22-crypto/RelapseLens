"""Single source of truth for ResistanceMap numbers, shapes, and topology.

All documentation (README.md, AGENT_ORCHESTRATOR.md, ARCHITECTURE.md) and all
assertions in code must render / import from this module. CI check
``scripts/check_docs_consistent.py`` fails the build if docs drift.

Rule: if a number appears in the README, it must either appear here or be
computed from values here.
"""
from __future__ import annotations

FACTS = {
    # ---- Protein-protein interaction network ---------------------------
    "ppi": {
        "source": "STRING v12",
        "raw_nodes": 19_566,
        "raw_edges": 11_938_498,
        # The MM subnet is what the GNN is ACTUALLY trained on.
        "mm_subnet_nodes": 7_853,
        "mm_subnet_edges": 460_212,
        "edge_direction": "undirected",   # confirmed from STRING schema
    },

    # ---- Model hyperparameters -----------------------------------------
    "vae": {
        "latent_dim": 64,
        "pretrain_epochs": 200,
        "finetune_epochs": 100,
        "pretrain_cohort": "CCLE_pan_cancer",
        "finetune_cohort": "CCLE_hematologic",
    },
    "trajectory": {
        "ode_solver": {
            "name": "dopri5",
            "family": "Dormand-Prince Runge-Kutta (4,5)",  # NOT Adams-Bashforth
            "source": "torchdiffeq",
        },
        "horizons_months": [3, 6, 12],
    },
    "protein_net": {
        "conv": "sage",              # GraphSAGE w/ neighbour sampling (P2 §9.1)
        "n_layers": 4,
        "num_neighbors": [25, 10, 10, 5],
        "in_dim": 1280,              # ESM-2 t33 output
        "hidden_dim": 256,
    },
    "fusion": {
        "kind": "cross_attention",
        "hidden_dim": 256,           # RECONCILED: was 256 in README, 128 in orchestrator
        "n_heads": 4,
    },

    # ---- Pipeline topology ---------------------------------------------
    # 9 stages. L2 has two parallel agents (VAEPretrain || ESM2Embed).
    "layers": [
        "L0_DataValidation",
        "L1_DataPrep",
        "L2_VAEPretrain",
        "L2_ESM2Embed",
        "L3_VAEFinetune",
        "L4_Trajectory",
        "L5_ProteinNet",
        "L6_Fusion",
        "L7_Landscape",
        "L8_Validation",
    ],

    # ---- Data budget ---------------------------------------------------
    # Sizes are uncompressed upstream; the release may ship a subsample.
    "data_sources": {
        "CCLE_proteomics":    {"size_gb": 1.2, "public": True},
        "CCLE_epigenomics":   {"size_gb": 2.5, "public": True},
        "STRING_PPI_v12":     {"size_gb": 0.4, "public": True},
        "GDSC":               {"size_gb": 0.15, "public": True},
        "CTRPv2":             {"size_gb": 0.3, "public": False, "access": "registered"},
        "GSE124310":          {"size_gb": 5.0, "public": True},
        "GSE271107":          {"size_gb": 8.0, "public": True},
        "MMRF_CoMMpass":      {"size_gb": 50.0, "public": False,
                               "access": "dbGaP DAR required"},
    },

    # ---- Security / audit framing (HONEST LABELS) ----------------------
    "verification": {
        "label": "Per-stage SHA256 content-hash chain for reproducibility",
        # Do NOT claim NIST SP 800-207 compliance -- see speculative review.
        "nist_zta_compliant": False,
    },
}


def total_raw_data_gb() -> float:
    """Sum of documented upstream raw data. Compare against release bundle."""
    return sum(s["size_gb"] for s in FACTS["data_sources"].values())


def public_only_data_gb() -> float:
    """Size of the subset that can legally be redistributed."""
    return sum(s["size_gb"] for s in FACTS["data_sources"].values() if s["public"])
