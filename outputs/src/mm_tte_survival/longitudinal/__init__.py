"""Clinical-anchored latent first-hitting-time (FHT) teacher.

Timescale substitution: the resistance/relapse latent state is *observed* through serially
available clinical biomarkers (M-protein, FLC, routine labs) rather than a serial molecular
trajectory. A scalar disease-burden readout s_t = w_cᵀ z_t is modelled as drifting Brownian
motion; its first passage to a barrier a_c is the calendar-time TT2L event, linked via the
inverse-Gaussian FHT already implemented in training/losses.py.

This is a legitimate Lee–Whitmore latent-crossing model ONLY because the crossed states are
clinical and repeatedly observed. The molecular-trajectory claim stays blocked (see
endpoints.MIN_MOLECULAR_PAIRS_FOR_TRAJECTORY).
"""
from .crossing_head import (
    CrossingHead, MonotoneReadout, fht_params_from_crossing,
    crossing_log_survival, crossing_log_pdf, crossing_nll,
    crossing_risk_score, never_cross_prob)

__all__ = ["CrossingHead", "MonotoneReadout", "fht_params_from_crossing",
           "crossing_log_survival", "crossing_log_pdf", "crossing_nll",
           "crossing_risk_score", "never_cross_prob"]
