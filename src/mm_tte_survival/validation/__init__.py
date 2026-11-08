"""Subtype-label validation: external real-FISH (GEO), internal cross-modality
concordance, label-noise robustness, and a FISH-ready harness.

These modules answer the only question that licenses any subtype claim: *are the
sequencing-inferred subtype labels trustworthy?* No FISH ground truth exists for
the open CoMMpass cohort (MMRF seqFISH is controlled-access), so trust is
established with a layered, strictly-open-data stack and the residual limitation
is stated plainly. Nothing here fabricates a FISH call.
"""
