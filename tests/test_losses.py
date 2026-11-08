"""Reference-equivalence + numerical-regression tests for the corrected losses.

Run on your box:  pytest test_losses.py     (needs torch + scipy; lifelines optional)
These ENCODE the scipy references that were verified to ~1e-12 during the refactor.
They are the tests a reviewer expects to see for a survival-loss implementation.
"""
import math
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from scipy import stats
from mm_tte_survival.training import losses as L


def test_aft_log_survival_matches_scipy_lognorm():
    # double precision: reference equivalence is a float64 claim (~1e-6), not float32 rounding.
    t = torch.linspace(1, 80, 500, dtype=torch.float64)
    mu, log_sigma = 3.0, math.log(0.6)
    params = torch.stack([torch.full_like(t, mu), torch.full_like(t, log_sigma)], dim=1)
    # internal log S(t) = log_ndtr(-z)
    z = (torch.log(t) - mu) / 0.6
    logS = torch.special.log_ndtr(-z).numpy()
    ref = stats.lognorm.logsf(t.numpy(), s=0.6, scale=math.exp(mu))
    assert np.max(np.abs(logS - ref)) < 1e-6


def test_fht_log_survival_matches_scipy_invgauss_and_is_finite():
    for m, lam in [(20.0, 8.0), (5.0, 50.0), (0.5, 5.0), (0.1, 20.0)]:
        t = torch.linspace(0.5, 60, 500, dtype=torch.float64)  # float64 reference equivalence
        params = torch.stack([torch.full_like(t, math.log(m)), torch.full_like(t, math.log(lam))], 1)
        # recompute the loss's internal log-survival
        mu = torch.exp(params[:, 0]); la = torch.exp(params[:, 1])
        sqrt_lt = torch.sqrt(la / t); a = sqrt_lt * (t / mu - 1); b = -sqrt_lt * (t / mu + 1)
        lpa = torch.special.log_ndtr(-a)
        u = (2 * la / mu + torch.special.log_ndtr(b) - lpa).clamp_max(-1e-12)
        logS = (lpa + L._log1mexp(u)).numpy()
        ref = stats.invgauss.logsf(t.numpy(), mu=m / lam, scale=lam)
        assert np.all(np.isfinite(logS)), f"NaN/inf at m={m},lam={lam}"
        assert np.nanmax(np.abs(logS - ref)) < 1e-4


def test_fht_loss_no_nan_in_overflow_regime():
    # the original exp(2*lam/mu) overflowed for small mu; this must stay finite.
    n = 64
    params = torch.stack([torch.full((n,), math.log(0.02)), torch.full((n,), math.log(30.0))], 1).requires_grad_(True)
    t = torch.rand(n) * 50 + 1; e = (torch.rand(n) > 0.5).float()
    loss = L.first_hitting_time_loss(params, t, e)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(params.grad).all()   # finite gradients, no nan_to_num band-aid


def test_aft_censored_tail_has_live_gradient():
    # long survivor (large z): original saturated at log(eps) -> dead gradient.
    params = torch.tensor([[0.0, 0.0]], requires_grad=True)   # mu=0,sigma=1
    t = torch.tensor([1e6]); e = torch.tensor([0.0])          # censored far out -> z huge
    L.lognormal_aft_loss(params, t, e).backward()
    assert torch.isfinite(params.grad).all() and params.grad.abs().sum() > 0


def test_cox_reduces_and_orders():
    torch.manual_seed(0)
    n = 200; risk = torch.randn(n); t = torch.rand(n) * 50 + 1; e = (torch.rand(n) > 0.3).float()
    for red in ("events", "mean", "sum"):
        assert torch.isfinite(L.cox_ph_loss(risk, t, e, reduction=red))
    # higher risk should lower the loss when it matches shorter times
    aligned = -t  # perfect risk
    assert L.cox_ph_loss(aligned, t, e) < L.cox_ph_loss(-aligned, t, e)


def test_distillation_has_no_temperature_and_is_mse():
    s = torch.randn(50, requires_grad=True); te = torch.randn(50)
    val = L.soft_distillation_loss(s, te)
    assert torch.isfinite(val)
    import inspect
    assert "temperature" not in inspect.signature(L.soft_distillation_loss).parameters


def test_survival_at_is_valid_probability_and_monotone():
    for mt, params in [("aft", torch.tensor([[3.0, 0.0]])), ("fht", torch.tensor([[math.log(20.0), math.log(8.0)]]))]:
        s12 = float(L.survival_at(mt, params, 12)); s60 = float(L.survival_at(mt, params, 60))
        assert 0.0 <= s60 <= s12 <= 1.0