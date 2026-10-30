"""Pytest configuration and shared fixtures for both test tiers."""
import os
import sys
import pytest

# Ensure literature-classifier is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.nli import StubScorer, DebertaNLIScorer
from app.text_utils import init_punkt

@pytest.fixture(scope="session", autouse=True)
def prepare_nltk():
    init_punkt()

@pytest.fixture
def stub_scorer():
    return StubScorer()

@pytest.fixture(scope="session")
def real_scorer():
    """Session-scoped real DeBERTa model for Tier 2 model tests."""
    return DebertaNLIScorer()
