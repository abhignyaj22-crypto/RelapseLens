"""Phase-A falsification gate for the latent-crossing head.

The state-space/EM (Stage 1) is only worth building if the crossing math is correct. These
tests pin it three ways: (1) the (gap, drift, sigma) -> inverse-Gaussian mapping matches
scipy.stats.invgauss; (2) it matches a Monte-Carlo Brownian first-passage simulation;
(3) the readout is monotone so the barrier/weight sign is identified.
"""
import math
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from scipy import stats

from mm_tte_survival.longitudinal.crossing_head import (
    CrossingHead, MonotoneReadout, fht_params_from_crossing)
from mm_tte_survival.training.losses import survival_at


def _closed_form_sf(t, mean, shape):
    # IG(mean, shape) survival via scipy: invgauss(mu=mean/shape, scale=shape).
    return stats.invgauss.sf(t, mu=mean / shape, scale=shape)


@pytest.mark.parametrize("gap,drift,sigma", [(2.0, 0.15, 0.5), (1.0, 0.3, 0.8), (3.0, 0.1, 0.4)])
def test_crossing_matches_scipy_invgauss(gap, drift, sigma):
    params = fht_params_from_crossing(torch.tensor([gap]), torch.tensor([drift]),
                                      torch.tensor([sigma])).double()
    mean, shape = gap / drift, gap * gap / (sigma * sigma)
    for h in (3.0, 6.0, 12.0, 24.0, 48.0):
        got = float(survival_at("fht", params, h)[0])
        ref = float(_closed_form_sf(h, mean, shape))
        assert abs(got - ref) < 1e-6, (h, got, ref)


def test_crossing_matches_monte_carlo_first_passage():
    gap, drift, sigma = 2.0, 0.2, 0.6
    rng = np.random.default_rng(0)
    n, dt, steps = 40000, 0.05, 4000
    x = np.zeros(n)
    hit = np.full(n, np.nan)
    for k in range(1, steps + 1):
        x += drift * dt + sigma * math.sqrt(dt) * rng.standard_normal(n)
        newly = np.isnan(hit) & (x >= gap)
        hit[newly] = k * dt
    params = fht_params_from_crossing(torch.tensor([gap]), torch.tensor([drift]),
                                      torch.tensor([sigma])).double()
    for h in (6.0, 12.0, 24.0):
        emp = float(np.mean((hit > h) | np.isnan(hit)))
        cf = float(survival_at("fht", params, h)[0])
        assert abs(emp - cf) < 0.03, (h, emp, cf)


def test_monotone_readout_weights_nonnegative_and_increasing():
    torch.manual_seed(0)
    head = MonotoneReadout(latent_dim=4)
    with torch.no_grad():
        head.raw_w.copy_(torch.randn(4) * 3.0)  # arbitrary raw weights, incl. negatives
    assert torch.all(head.weight >= 0.0)
    z = torch.zeros(1, 4)
    with torch.no_grad():
        base = head(z)
        bumped = head(z + torch.tensor([[1.0, 1.0, 1.0, 1.0]]))
    assert float(bumped) >= float(base)  # increasing latent => non-decreasing burden


def test_crossing_head_barrier_positive_and_gap_shrinks_survival():
    head = CrossingHead(n_strata=1, init_barrier=2.0)
    s0 = torch.tensor([0.0, 1.5])          # patient 2 is closer to the barrier
    drift = torch.tensor([0.2, 0.2])
    sigma = torch.tensor([0.5, 0.5])
    with torch.no_grad():
        params = head(s0, drift, sigma).double()
    s_near = float(survival_at("fht", params[1:2], 12.0))
    s_far = float(survival_at("fht", params[0:1], 12.0))
    assert float(head.barrier().detach()) > 0.0
    assert s_near < s_far  # smaller gap => hits sooner => lower survival
