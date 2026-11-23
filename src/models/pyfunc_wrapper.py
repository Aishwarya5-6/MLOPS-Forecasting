"""Wrap any ForecastModel as an MLflow pyfunc so every model loads and predicts the same way.

Input:  DataFrame with columns `timestamp` (hourly) and `pm25`: the recent history.
Params: {"horizon": 24}
Output: DataFrame with columns `timestamp` and `forecast`, for the hours after the history.

    model = mlflow.pyfunc.load_model("models:/pm25-forecaster@champion")
    model.predict(history_df, params={"horizon": 24})
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature

from src.models.base import ForecastModel

PIP_PACKAGES = [
    "mlflow", "pandas", "numpy", "scikit-learn", "statsmodels", "xgboost", "holidays",
    "joblib", "pyyaml", "python-dotenv",
]  # fmt: skip


class ForecastPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model: ForecastModel = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input, params=None):
        horizon = int((params or {}).get("horizon", 24))
        hist = (
            model_input.assign(timestamp=pd.to_datetime(model_input["timestamp"]))
            .set_index("timestamp")["pm25"]
            .astype(float)
            .sort_index()
            .asfreq("h")  # regular hourly grid; gaps become NaN
        )
        forecast = self.model.predict(hist, horizon)
        times = pd.date_range(hist.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
        return pd.DataFrame({"timestamp": times, "forecast": forecast})


def history_frame(y: pd.Series, hours: int) -> pd.DataFrame:
    tail = y.iloc[-hours:]
    return pd.DataFrame({"timestamp": tail.index, "pm25": tail.to_numpy(dtype=float)})


def log_pyfunc(model: ForecastModel, example: pd.DataFrame, tmp_dir: Path, horizon: int):
    """Log `model` as a pyfunc in the active MLflow run. Returns the ModelInfo."""
    path = Path(tmp_dir) / "model.joblib"
    joblib.dump(model, path)
    wrapper = ForecastPyfunc()
    wrapper.model = model
    output = wrapper.predict(None, example, {"horizon": horizon})
    signature = infer_signature(example, output, params={"horizon": horizon})
    root = Path(__file__).resolve().parents[1]  # the src/ package
    return mlflow.pyfunc.log_model(
        name="model",
        python_model=wrapper,
        artifacts={"model": str(path)},
        code_paths=[str(root)],
        signature=signature,
        input_example=example,
        pip_requirements=[
            f"{p}=={version(p)}" for p in PIP_PACKAGES + getattr(model, "extra_pip", [])
        ],
    )
