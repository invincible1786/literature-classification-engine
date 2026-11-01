import sys
import os
import pytest

# Ensure literature-classifier is in Python path for test discovery
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (loads full neural weights)")

