import importlib
from pathlib import Path


def test_project_scaffold_exists():
    assert Path("pyproject.toml").exists()
    assert Path("src/credit_domino").is_dir()


def test_credit_domino_is_importable():
    mod = importlib.import_module("credit_domino")
    assert hasattr(mod, "__version__")
