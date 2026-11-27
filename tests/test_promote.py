"""promote.py: the decision rule, plus the full registry flow against a throwaway MLflow."""

import math

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from src import promote
from src.promote import decide


# --------------------------------------------------------------------------- decision rule
def test_no_champion_promotes():
    d = decide(12.0, None)
    assert d.promote and d.improvement is None


def test_exactly_two_percent_promotes():
    d = decide(9.8, 10.0)
    assert d.promote
    assert d.improvement == pytest.approx(0.02)


def test_just_under_two_percent_rejects():
    d = decide(9.81, 10.0)
    assert not d.promote
    assert "not enough improvement" in d.reason


def test_worse_challenger_rejects():
    d = decide(11.0, 10.0)
    assert not d.promote and d.improvement < 0


def test_nan_challenger_rejects():
    assert not decide(math.nan, 10.0).promote


def test_threshold_is_configurable():
    assert decide(9.95, 10.0, min_improvement=0.0).promote
    assert not decide(9.5, 10.0, min_improvement=0.1).promote


# --------------------------------------------------------------------------- registry flow
@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Throwaway SQLite MLflow; scorer returns a fixed RMSE per registered run."""
    from src.tracking import data_version

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "promote-test")
    monkeypatch.setattr(promote, "MODEL_NAME", "test-forecaster")
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("promote-test", artifact_location=str(tmp_path / "artifacts"))
    mlflow.set_experiment("promote-test")
    client = MlflowClient(uri)
    current = data_version()["prepared_data_version"]
    rmse_by_run: dict[str, float] = {}

    def make_run(holdout_rmse: float, data: str = current) -> str:
        from src.models.baseline import NaiveModel
        from src.models.pyfunc_wrapper import history_frame, log_pyfunc

        y = pd.Series(np.arange(48.0), index=pd.date_range("2026-01-01", periods=48, freq="h"))
        with mlflow.start_run() as run:
            mlflow.set_tags({"prepared_data_version": data, "model_type": "naive"})
            mlflow.log_metric("rmse", holdout_rmse)
            log_pyfunc(NaiveModel().fit(y), history_frame(y, 24), tmp_path, horizon=24)
        rmse_by_run[run.info.run_id] = holdout_rmse
        return run.info.run_id

    def scorer(model_uri: str) -> dict:
        if model_uri.startswith("runs:/"):
            run_id = model_uri.split("/")[1]
        else:
            alias = model_uri.rsplit("@", 1)[-1]
            run_id = client.get_model_version_by_alias("test-forecaster", alias).run_id
        r = rmse_by_run[run_id]
        return {"rmse": r, "mae": r * 0.8, "smape": 30.0, "window": "w", "n_origins": 29}

    yield client, make_run, scorer
    mlflow.set_tracking_uri("http://127.0.0.1:5000")


def _champion(client):
    return client.get_model_version_by_alias("test-forecaster", "champion")


def test_promotion_flow_and_rollback(registry):
    client, make_run, scorer = registry

    # 1) No champion yet: first challenger wins.
    r1 = make_run(10.0)
    assert promote.cmd_challenger(r1, scorer=scorer).promote
    v1 = _champion(client)
    assert str(v1.version) == "1" and v1.tags["previous_champion"] == "none"
    assert v1.tags["holdout_rmse"] == "10.0000"

    # 2) Only 1% better: rejected, champion unchanged, challenger tagged with the reason.
    r2 = make_run(9.9)
    assert not promote.cmd_challenger(r2, scorer=scorer).promote
    assert str(_champion(client).version) == "1"
    v2 = client.get_model_version_by_alias("test-forecaster", "challenger")
    assert str(v2.version) == "2" and "not enough improvement" in v2.tags["rejected_reason"]

    # 3) 5% better: promoted, previous champion recorded, @challenger cleared.
    r3 = make_run(9.5)
    assert promote.cmd_challenger(r3, scorer=scorer).promote
    v3 = _champion(client)
    assert str(v3.version) == "3" and v3.tags["previous_champion"] == "1"
    assert "challenger" not in client.get_registered_model("test-forecaster").aliases

    # 4) Rollback: one command puts v1 back.
    assert promote.cmd_rollback() == "1"
    assert str(_champion(client).version) == "1"
    assert client.get_model_version("test-forecaster", "3").tags["rolled_back_to"] == "1"

    # 5) v1 has no earlier champion, so a second rollback refuses.
    with pytest.raises(SystemExit):
        promote.cmd_rollback()


def test_dry_run_changes_nothing(registry):
    client, make_run, scorer = registry
    promote.cmd_challenger(make_run(10.0), scorer=scorer)
    d = promote.cmd_challenger(make_run(5.0), dry_run=True, scorer=scorer)
    assert d.promote
    assert str(_champion(client).version) == "1"
    assert len(client.search_model_versions("name='test-forecaster'")) == 1


def test_rejects_run_trained_on_other_data(registry):
    _, make_run, scorer = registry
    with pytest.raises(SystemExit, match="Retrain"):
        promote.cmd_challenger(make_run(5.0, data="stale-hash"), scorer=scorer)


def test_reusing_a_run_does_not_create_a_new_version(registry):
    client, make_run, scorer = registry
    r1 = make_run(10.0)
    promote.cmd_challenger(r1, scorer=scorer)
    with pytest.raises(SystemExit, match="already @champion"):
        promote.cmd_challenger(r1, scorer=scorer)
    assert len(client.search_model_versions("name='test-forecaster'")) == 1
