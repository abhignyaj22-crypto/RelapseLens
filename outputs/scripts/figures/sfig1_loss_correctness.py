#!/usr/bin/env python
"""Supplement Figure — survival-loss correctness (pairs with tests/test_losses.py).

A. Reference equivalence: corrected AFT (log-normal) and FHT (inverse-Gaussian)
   log-survival vs scipy closed forms — overlap to ~1e-6/1e-4 (annotated max|err|).
B. The censored-tail fix: the old log(clamp(1−Φ(z), eps)) SATURATES at log(eps)≈−18.4
   for z≳6 (flat = dead gradient for every long survivor), while the corrected
   log_ndtr(−z) stays live out to z≈40. Most of a heavily-censored MM cohort lives in
   exactly that tail.

This figure plots reference mathematics (closed-form vs closed-form), not a fitted
model result, so it does not violate the Fragility-1 "figures fit nothing" rule.
"""
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import COLORS, save, plt

import torch
from scipy import stats
from mm_tte_survival.training import losses as L

EPS = 1e-8


def _panel_reference(ax):
    # AFT: log S(t) = log_ndtr(-(log t - mu)/sigma) vs scipy lognorm.logsf
    t = torch.linspace(1, 80, 400, dtype=torch.float64)
    mu, sigma = 3.0, 0.6
    z = (torch.log(t) - mu) / sigma
    logS_aft = torch.special.log_ndtr(-z).numpy()
    ref_aft = stats.lognorm.logsf(t.numpy(), s=sigma, scale=math.exp(mu))
    err_aft = np.max(np.abs(logS_aft - ref_aft))

    # FHT: inverse-Gaussian log-survival via log_ndtr + log1mexp vs scipy invgauss.logsf
    m, lam = 20.0, 8.0
    tt = torch.linspace(0.5, 60, 400, dtype=torch.float64)
    sqrt_lt = torch.sqrt(torch.tensor(lam) / tt)
    a = sqrt_lt * (tt / m - 1.0); b = -sqrt_lt * (tt / m + 1.0)
    lpa = torch.special.log_ndtr(-a)
    u = (2 * lam / m + torch.special.log_ndtr(b) - lpa).clamp_max(-1e-12)
    logS_fht = (lpa + L._log1mexp(u)).numpy()
    ref_fht = stats.invgauss.logsf(tt.numpy(), mu=m / lam, scale=lam)
    err_fht = np.nanmax(np.abs(logS_fht - ref_fht))

    ax.plot(t.numpy(), ref_aft, color=COLORS["clinical"], lw=4, alpha=0.35, label="AFT scipy ref")
    ax.plot(t.numpy(), logS_aft, color=COLORS["clinical"], lw=1.2, label="AFT ours")
    ax.plot(tt.numpy(), ref_fht, color=COLORS["omics"], lw=4, alpha=0.35, label="FHT scipy ref")
    ax.plot(tt.numpy(), logS_fht, color=COLORS["omics"], lw=1.2, label="FHT ours")
    ax.set_xlabel("t", fontsize=8.5); ax.set_ylabel("log S(t)", fontsize=8.5)
    ax.set_title(f"A. Reference equivalence (max|err| AFT {err_aft:.1e}, FHT {err_fht:.1e})", fontsize=9)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9)


def _panel_tail(ax):
    z = np.linspace(0, 12, 400)
    corrected = torch.special.log_ndtr(torch.tensor(-z)).numpy()         # stays live
    old_clamp = np.log(np.clip(1.0 - stats.norm.cdf(z), EPS, None))      # saturates at log(eps)
    ax.plot(z, corrected, color=COLORS["confirmed"], lw=1.8, label="corrected  log_ndtr(−z)")
    ax.plot(z, old_clamp, color=COLORS["blocked"], lw=1.8, ls="--", label="old  log(clamp(1−Φ, eps))")
    ax.axhline(math.log(EPS), color=COLORS["blocked"], lw=0.7, ls=":")
    ax.axvspan(6, 12, color=COLORS["blocked"], alpha=0.08)
    ax.text(9, math.log(EPS) + 3, "dead gradient\n(z≳6)", color=COLORS["blocked"], fontsize=7.5, ha="center")
    ax.set_xlabel("z = (log t − μ)/σ  (long survivor → large z)", fontsize=8.5)
    ax.set_ylabel("log S", fontsize=8.5)
    ax.set_title("B. Censored-tail fix: old loss saturates, gradient dies", fontsize=9)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    _panel_reference(axes[0]); _panel_tail(axes[1])
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.text(0.01, -0.03, "Pairs with tests/test_losses.py (reference-equivalence + "
             "live-censored-gradient + overflow-finiteness). Illustrative closed-form math.",
             fontsize=6.8, color="#555555")
    fig.tight_layout()
    save(fig, "sfig1_loss_correctness")


if __name__ == "__main__":
    main()
