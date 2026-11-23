import pytest

from src.train import load_config, parse_args


def test_parse_args_model():
    args = parse_args(["--model", "dummy"])
    assert args.model == "dummy"
    assert args.config is None


def test_parse_args_rejects_unknown_model():
    with pytest.raises(SystemExit):
        parse_args(["--model", "nope"])


def test_load_config_missing_file_returns_empty(tmp_path):
    assert load_config("dummy", tmp_path / "missing.yaml") == {}
