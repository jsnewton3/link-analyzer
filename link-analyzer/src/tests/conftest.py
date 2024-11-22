import pytest
from flask_app_factory import create_app
import json
import glob
import os
TEST_JSONS_PATH = "/link-analyzer/src/network-monitor-server/tests/test_jsons"
@pytest.fixture
def client():
    app = create_app()
    yield app.test_client()

@pytest.fixture
def test_jsons():
    test_jsons = {}
    for file in glob.glob(os.path.join(TEST_JSONS_PATH, '*.json')):
        with open(os.path.join(TEST_JSONS_PATH, file)) as f:
            key = os.path.split(file)[1]
            test_jsons[key] = json.load(f)
    yield test_jsons
