"""Promotion study: stats, policies and the replay loop, on synthetic data with known answers."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.study.cache import forecast_segment, make_schedule
from src.study.policies import DieboldMariano, Never, Threshold, daily_loss
from src.study.replay import initial_champion, replay_oracle, replay_policy
from src.study.stats import bootstrap_rmse, ci, dm_test

H = 24


# --------------------------------------------------------------------------- statistics
def test_dm_matches_hand_calculation():
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # mean 3, population variance 2 -> DM = 3 / sqrt(2/5); HLN factor sqrt((5+1-2)/5)
    expected = 3 / np.sqrt(2 / 5) * np.sqrt(4 / 5)
    stat, p = dm_test(d)
    assert stat == pytest.approx(expected)
    assert p == pytest.approx(2 * stats.t(df=4).sf(expected))
    assert dm_test(-d, alternative="less")[1] == pytest.approx(stats.t(df=4).cdf(-expected))


def test_dm_degenerate_inputs():
    assert dm_test([1.0])[1] == 1.0
    assert dm_test([2.0, 2.0, 2.0])[1] == 1.0  # zero variance
    assert dm_test([np.nan, 1.0, 2.0, 3.0])[0] == pytest.approx(dm_test([1.0, 2.0, 3.0])[0])


def test_bootstrap_ci_covers_true_rmse():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 3.0, size=(52, 168))  # true RMSE 3
    sse, n = {"a": (e**2).sum(axis=1)}, {"a": np.full(52, 168.0)}
    lo, hi = ci(bootstrap_rmse(sse, n, 2000, 1)["a"])
    assert lo < 3.0 < hi and hi - lo < 0.2


# --------------------------------------------------------------------------- policies
def test_threshold_policy_matches_promote_rule():
    champ, chall = np.full((7, H), 100.0), np.full((7, H), 96.0)  # RMSE 10 vs 9.80 = 2.0%
    assert Threshold(0.02)(champ, chall)
    assert not Threshold(0.05)(champ, chall)
    assert Threshold(0.0)(champ, chall)
    assert not Never()(champ, chall)
    assert Threshold(0.02).name == "tau2" and Threshold(0.10).name == "tau10"


def test_dm_policy_needs_consistent_evidence():
    rng = np.random.default_rng(1)
    base = rng.uniform(50, 150, size=(30, H))
    assert DieboldMariano()(base, base * 0.8)  # 20% better every day -> significant
    noisy = base * rng.uniform(0.7, 1.3, size=(30, 1))  # same on average
    assert not DieboldMariano()(base, noisy)


def test_daily_loss_skips_missing_hours():
    e = np.array([[1.0, 3.0, np.nan], [np.nan, np.nan, np.nan]])
    out = daily_loss(e)
    assert out[0] == 2.0 and np.isnan(out[1])


# --------------------------------------------------------------------------- replay
def _setup(err_by_model: dict, n_days: int):
    origins = pd.date_range("2025-01-01", periods=n_days, freq="D")
    start = origins[30]
    decisions = [start + pd.Timedelta(days=7 * k) for k in range((n_days - 30) // 7)]
    return origins, err_by_model, decisions, start


def test_strictly_better_model_is_found_and_kept():
    rng = np.random.default_rng(2)
    n = 30 + 7 * 20
    err = {"good": rng.normal(0, 1, (n, H)) ** 2, "bad": rng.normal(0, 3, (n, H)) ** 2}
    origins, err, decisions, start = _setup(err, n)
    init = initial_champion(origins, err, start, 30)
    assert init == "good"
    for policy in (Threshold(0.0), Threshold(0.1), DieboldMariano()):
        r = replay_policy(policy, 7, origins, err, decisions, "bad")  # start from the wrong one
        assert r.champions[-1] == "good" and r.swaps == 1 and r.harmful == 0
    oracle = replay_oracle(origins, err, decisions)
    r = replay_policy(Threshold(0.02), 30, origins, err, decisions, init)
    assert r.swaps == 0 and r.rmse == pytest.approx(oracle.rmse)


def test_equal_models_zero_threshold_churns():
    """Five identical-quality models: any swap is luck, so tau0 swaps often, ~half harmful."""
    rng = np.random.default_rng(3)
    n = 30 + 7 * 52
    err = {f"m{i}": rng.normal(0, 1, (n, H)) ** 2 for i in range(5)}
    origins, err, decisions, _ = _setup(err, n)
    r0 = replay_policy(Threshold(0.0), 7, origins, err, decisions, "m0")
    r10 = replay_policy(Threshold(0.10), 7, origins, err, decisions, "m0")
    rdm = replay_policy(DieboldMariano(), 7, origins, err, decisions, "m0")
    assert r0.swaps > 30
    assert r10.swaps < r0.swaps / 2
    assert rdm.swaps < r0.swaps / 2
    assert 0.3 < r0.harmful / r0.swaps < 0.7


def test_skip_when_validation_data_missing():
    n = 30 + 7 * 4
    err = {"a": np.ones((n, H)), "b": np.full((n, H), 0.25)}
    err["a"][:, :] = np.nan
    err["b"][:, :] = np.nan
    origins, err, decisions, _ = _setup(err, n)
    r = replay_policy(Threshold(0.0), 7, origins, err, decisions, "a")
    assert r.skipped == len(decisions) and r.swaps == 0


# --------------------------------------------------------------------------- cache leakage
class Spy:
    name = "spy"

    def __init__(self):
        self.fit_end, self.seen = None, []

    def fit(self, train):
        self.fit_end = train.index[-1]
        return self

    def predict(self, history, horizon):
        self.seen.append(history.index[-1])
        return np.zeros(horizon)


def test_cache_never_sees_the_future():
    idx = pd.date_range("2025-01-01", periods=24 * 60, freq="h")
    y = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    refit = pd.Timestamp("2025-02-01")
    origins = list(pd.date_range(refit, periods=5, freq="D"))
    spy = Spy()
    rows = forecast_segment(spy, y, refit, origins, H)
    assert spy.fit_end < refit
    assert all(seen < o for seen, o in zip(spy.seen, origins, strict=True))
    assert rows[0][4] == y[refit]  # actual stored for step 1 is the origin hour


def test_schedule_layout():
    idx = pd.date_range("2024-01-01", "2025-06-30 23:00", freq="h")
    study = {"replay_days": 365, "max_window_days": 30, "refit_every_days": 30}
    s = make_schedule(idx, study)
    assert s.replay_end == pd.Timestamp("2025-06-30")
    assert (s.replay_end - s.replay_start).days == 364
    assert s.cache_start == s.replay_start - pd.Timedelta(days=30)
    assert s.refits[0] == s.cache_start and len(s.origins) == 395
