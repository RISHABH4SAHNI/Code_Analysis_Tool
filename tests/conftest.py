import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def fake_api_keys(monkeypatch):
    """Service constructors require keys; tests never call the real APIs"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-perplexity-key")
