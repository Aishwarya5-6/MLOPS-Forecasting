import pandas as pd
import pytest

from src.data import make_folds

H = pd.Timedelta(hours=1)
INDEX = pd.date_range("2022-08-04 05:00", "2026-11-23 04:00", freq="h")
KW = dict(
    horizon=24, n_folds=5, test_hours=720, step_hours=1752, holdout_hours=720, origin_every=24
)


@pytest.fixture
def split():
    return make_folds(INDEX, **KW)


def test_number_of_folds(split):
    folds, _ = split
    assert len(folds) == 5


def test_test_always_after_train(split):
    folds, holdout = split
    for f in [*folds, holdout]:
        assert f.train_end < f.test_start
        assert f.test_start - f.train_end == H  # no gap, no overlap


def test_test_windows_do_not_overlap_and_are_ordered(split):
    folds, _ = split
    for a, b in zip(folds, folds[1:], strict=False):
        assert a.test_end < b.test_start


def test_expanding_window(split):
    folds, _ = split
    ends = [f.train_end for f in folds]
    assert ends == sorted(ends)


def test_folds_never_touch_holdout(split):
    folds, holdout = split
    assert all(f.test_end < holdout.test_start for f in folds)
    assert holdout.test_end == INDEX[-1]
    assert holdout.test_start.hour == 0
    assert (holdout.test_end - holdout.test_start) / H + 1 <= 720


def test_origins_fit_inside_window(split):
    folds, holdout = split
    for f in [*folds, holdout]:
        assert f.origins[0] == f.test_start
        assert f.origins[-1] + 23 * H <= f.test_end
        assert all(o.hour == 0 for o in f.origins)


def test_too_many_folds_raises():
    short = pd.date_range("2026-01-01", periods=24 * 60, freq="h")
    with pytest.raises(ValueError):
        make_folds(short, **KW)
