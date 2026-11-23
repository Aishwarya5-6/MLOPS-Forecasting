"""LSTM tests run in a separate Python process.

torch and xgboost each ship their own OpenMP runtime (libomp). Loading both in one process
segfaults on macOS, whichever is imported first. The rest of the test suite imports xgboost,
so these tests start a fresh interpreter. Training runs never load both: train.py, the sweep
and the saved pyfunc each use a single model type per process.
"""

import subprocess
import sys
import textwrap

from src.config import ROOT

SETUP = """
import numpy as np, pandas as pd
from src.models.lstm import LSTMModel

def series(days, seed=0):
    rng = np.random.default_rng(seed)
    h = np.arange(days * 24)
    v = 30 + 10 * np.sin(2 * np.pi * h / 24) + rng.normal(0, 1, len(h))
    return pd.Series(v, index=pd.date_range("2025-01-01", periods=len(h), freq="h"))
"""


def _run(body: str) -> None:
    code = textwrap.dedent(SETUP) + textwrap.dedent(body)
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_lstm_fits_and_predicts():
    _run("""
        y = series(40)
        m = LSTMModel(window=48, hidden_size=16, max_epochs=3, train_days=40, train_stride=6)
        m.fit(y.iloc[:-24])
        pred = m.predict(y.iloc[:-24], 24)
        assert pred.shape == (24,)
        assert (pred >= 0).all()
        assert 1 <= m.epochs_trained_ <= 3
    """)


def test_lstm_is_reproducible_with_a_seed():
    _run("""
        y = series(30)
        kw = dict(window=48, hidden_size=8, max_epochs=2, train_days=30, train_stride=6, seed=7)
        a = LSTMModel(**kw).fit(y).predict(y, 24)
        b = LSTMModel(**kw).fit(y).predict(y, 24)
        np.testing.assert_allclose(a, b, rtol=1e-6)
    """)
