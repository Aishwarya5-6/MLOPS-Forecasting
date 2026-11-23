"""LSTM, direct multi-output: reads the last `window` hours, outputs all `horizon` steps at once."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models.base import ForecastModel


class _Net(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float, horizon: int):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )  # fmt: skip
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def _calendar(times: pd.DatetimeIndex) -> np.ndarray:
    """sin/cos of hour of day and day of week: shape (len(times), 4)."""
    h = 2 * np.pi * times.hour.to_numpy() / 24
    d = 2 * np.pi * times.day_of_week.to_numpy() / 7
    return np.stack([np.sin(h), np.cos(h), np.sin(d), np.cos(d)], axis=1)


class LSTMModel(ForecastModel):
    """Inputs per hour: scaled log1p(pm25) + 4 calendar terms. Scaling is fit on training data only.

    Training uses windows from the last `train_days` of the fold's training data. The last
    `val_fraction` of those windows (in time order) is held out for early stopping.
    """

    name = "lstm"
    extra_pip = ["torch"]

    def __init__(
        self,
        horizon: int = 24,
        window: int = 168,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 30,
        patience: int = 4,
        train_days: int = 730,
        train_stride: int = 2,
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        self.horizon = horizon
        self.window = window
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.train_days = train_days
        self.train_stride = train_stride
        self.val_fraction = val_fraction
        self.seed = seed
        self.net_: _Net | None = None
        self.mean_ = self.std_ = None
        self.epochs_trained_ = 0

    @property
    def min_history(self) -> int:
        return self.window

    def _inputs(self, series: pd.Series) -> np.ndarray:
        v = np.log1p(series.ffill().bfill().to_numpy(dtype=float))
        scaled = (v - self.mean_) / self.std_
        return np.column_stack([scaled, _calendar(series.index)]).astype(np.float32)

    def fit(self, train: pd.Series) -> LSTMModel:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        train = train.iloc[-self.train_days * 24 :]
        logv = np.log1p(train.ffill().bfill().to_numpy(dtype=float))
        self.mean_, self.std_ = float(logv.mean()), float(logv.std())
        X_all = self._inputs(train)
        y_all = X_all[:, 0]

        starts = np.arange(0, len(train) - self.window - self.horizon + 1, self.train_stride)
        X = np.stack([X_all[s : s + self.window] for s in starts])
        Y = np.stack([y_all[s + self.window : s + self.window + self.horizon] for s in starts])
        n_val = max(1, int(len(starts) * self.val_fraction))
        # Drop windows whose targets overlap the validation period (no leakage into early stopping).
        n_train = max(1, len(starts) - n_val - (self.window + self.horizon) // self.train_stride)
        Xt, Yt = torch.from_numpy(X[:n_train]), torch.from_numpy(Y[:n_train])
        Xv, Yv = torch.from_numpy(X[-n_val:]), torch.from_numpy(Y[-n_val:])

        net = _Net(X.shape[2], self.hidden_size, self.num_layers, self.dropout, self.horizon)
        opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        gen = torch.Generator().manual_seed(self.seed)
        best, best_state, bad, epochs = float("inf"), None, 0, 0
        for _ in range(self.max_epochs):
            epochs += 1
            net.train()
            for idx in torch.randperm(len(Xt), generator=gen).split(self.batch_size):
                opt.zero_grad()
                loss = loss_fn(net(Xt[idx]), Yt[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                val = loss_fn(net(Xv), Yv).item()
            if val < best - 1e-5:
                best, best_state, bad = val, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        self.net_, self.epochs_trained_ = net, epochs
        return self

    def predict(self, history: pd.Series, horizon: int) -> np.ndarray:
        if self.net_ is None:
            raise RuntimeError("fit() must be called before predict()")
        if horizon > self.horizon:
            raise ValueError(f"Model was trained for up to {self.horizon} steps ahead")
        x = torch.from_numpy(self._inputs(history.iloc[-self.window :]))[None]
        with torch.no_grad():
            out = self.net_(x)[0, :horizon].numpy()
        return np.clip(np.expm1(out * self.std_ + self.mean_), 0, None)

    def get_params(self) -> dict:
        return {
            k: getattr(self, k)
            for k in (
                "horizon", "window", "hidden_size", "num_layers", "dropout", "learning_rate",
                "batch_size", "max_epochs", "patience", "train_days", "train_stride",
                "val_fraction", "seed",
            )
        }  # fmt: skip

    def short_name(self) -> str:
        return f"lstm-h{self.hidden_size}-l{self.num_layers}-w{self.window}-lr{self.learning_rate}"
