"""pytest fixtures for glmocr tests"""

import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
RUN_INTEGRATION = os.getenv("GLMOCR_RUN_INTEGRATION", "0") == "1"
SERVER_URL = os.getenv("GLMOCR_SERVER_URL", "http://127.0.0.1:5002")
TEST_IMAGE = os.getenv("GLMOCR_TEST_IMAGE", "")
TEST_PDF = os.getenv("GLMOCR_TEST_PDF", "")
TIMEOUT_SECONDS = int(os.getenv("GLMOCR_TIMEOUT", "300"))


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires running server)",
    )


# ---------------------------------------------------------------------------
# Skip integration tests by default
# ---------------------------------------------------------------------------
def pytest_collection_modifyitems(config, items):
    if RUN_INTEGRATION:
        return
    skip_integration = pytest.mark.skip(
        reason="Integration test skipped. Set GLMOCR_RUN_INTEGRATION=1 to run."
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def server_url():
    """Return the server URL for integration tests."""
    return SERVER_URL


@pytest.fixture
def timeout_seconds():
    """Return timeout for requests."""
    return TIMEOUT_SECONDS


@pytest.fixture
def sample_image_path():
    """Return path to a sample test image if set via env."""
    if TEST_IMAGE and Path(TEST_IMAGE).exists():
        return Path(TEST_IMAGE)
    # Try default path
    default = Path(__file__).parent.parent.parent / "examples" / "source"
    if default.exists():
        images = list(default.glob("*.png")) + list(default.glob("*.jpg"))
        if images:
            return images[0]
    return None


@pytest.fixture
def sample_pdf_path():
    """Return path to a sample test PDF if set via env or available in repo."""
    if TEST_PDF and Path(TEST_PDF).exists():
        return Path(TEST_PDF)

    # Try default path
    default_dir = Path(__file__).parent.parent.parent / "examples" / "source"
    if default_dir.exists():
        pdfs = list(default_dir.glob("*.pdf"))
        if pdfs:
            return pdfs[0]

    return None


# PDF fixture was removed previously; can be re-added when appropriate.
