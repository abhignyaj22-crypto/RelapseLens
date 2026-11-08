"""ResistanceMap: Foundation model for predicting drug resistance trajectories in hematologic malignancies.

A multi-modal deep learning system combining:
  - Protein and epigenomic data (from cell lines and patient samples)
  - Single-cell transcriptomics (for sub-population heterogeneity)
  - Protein-protein interaction networks (ESM-2 embeddings)
  - Trajectory modeling (ODE-based resistance dynamics)
  - Temporal forecasting (3, 6, 12 month horizons)
"""

__version__ = "0.1.0"
__author__ = "Anthropic"
__license__ = "MIT"

# Honest-core cleanup: the aspirational breadth subpackages were removed. `resistancemap` now exposes only
# the thin goal seam (goal, cli, integration.core_bridge, inference.honest_api, evaluation.baselines), each
# imported directly by its consumer rather than eagerly here.
