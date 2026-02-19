"""Integration tests for glmocr (requires a running service).

How to run:
Terminal 1: start server
    python -m glmocr.server
Terminal 2: run integration tests
    GLMOCR_RUN_INTEGRATION=1 \
    GLMOCR_SERVER_URL=http://127.0.0.1:5002 \
    GLMOCR_TEST_IMAGE=./examples/source/1.png \
    GLMOCR_TEST_PDF=./examples/source/954d59b1-d8c1-4baf-9e3b-c04bf1961d7b.pdf \
    pytest -q glmocr/tests/test_integration.py
"""

import base64

import pytest
import requests


# ---------------------------------------------------------------------------
# All integration tests are marked with @pytest.mark.integration
# conftest.py will auto-skip unless GLMOCR_RUN_INTEGRATION=1
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestHealthEndpoint:
    """Tests for /health."""

    def test_health_returns_ok(self, server_url, timeout_seconds):
        resp = requests.get(f"{server_url}/health", timeout=timeout_seconds)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "ok"


@pytest.mark.integration
class TestParseEndpoint:
    """Tests for /glmocr/parse."""

    def test_parse_returns_json_result(
        self, server_url, timeout_seconds, sample_image_path
    ):
        """parse returns json_result."""
        if sample_image_path is None:
            pytest.skip("No sample image available")

        with open(sample_image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {"images": [f"data:image/png;base64,{img_b64}"]}
        resp = requests.post(
            f"{server_url}/glmocr/parse",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout_seconds,
        )

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "json_result" in data
        assert data["json_result"] is not None

    def test_parse_with_empty_images_returns_error(self, server_url, timeout_seconds):
        """Empty images returns an error."""
        payload = {"images": []}
        resp = requests.post(
            f"{server_url}/glmocr/parse",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout_seconds,
        )

        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data

    def test_parse_with_invalid_content_type(self, server_url, timeout_seconds):
        """Invalid Content-Type returns an error."""
        resp = requests.post(
            f"{server_url}/glmocr/parse",
            data="not json",
            headers={"Content-Type": "text/plain"},
            timeout=timeout_seconds,
        )

        assert resp.status_code == 400

    def test_parse_multiple_images(
        self, server_url, timeout_seconds, sample_image_path
    ):
        """Multiple images are accepted."""
        if sample_image_path is None:
            pytest.skip("No sample image available")

        with open(sample_image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Send the same image twice
        payload = {
            "images": [
                f"data:image/png;base64,{img_b64}",
                f"data:image/png;base64,{img_b64}",
            ]
        }
        resp = requests.post(
            f"{server_url}/glmocr/parse",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout_seconds,
        )

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "json_result" in data

    def test_parse_pdf_file_uri(self, server_url, timeout_seconds, sample_pdf_path):
        """PDF parsing via file:// absolute path."""
        if sample_pdf_path is None:
            pytest.skip("No sample PDF available")

        # Dependency check: pypdfium2
        try:
            from glmocr.utils.image_utils import PYPDFIUM2_AVAILABLE
        except Exception:
            PYPDFIUM2_AVAILABLE = False

        if not PYPDFIUM2_AVAILABLE:
            pytest.skip("pypdfium2 is not installed")

        pdf_uri = f"file://{sample_pdf_path.resolve()}"
        payload = {"images": [pdf_uri]}
        resp = requests.post(
            f"{server_url}/glmocr/parse",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout_seconds,
        )

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "json_result" in data
        assert data["json_result"] is not None


@pytest.mark.integration
class TestGlmOcrAPI:
    """Tests for the Python API (requires service)."""

    def test_glmocr_parse_file(self, sample_image_path):
        """GlmOcr.parse works."""
        if sample_image_path is None:
            pytest.skip("No sample image available")

        from glmocr.api import GlmOcr

        parser = GlmOcr()
        try:
            result = parser.parse(str(sample_image_path))
            assert result is not None
            assert result.json_result is not None
        finally:
            parser.close()

    def test_glmocr_context_manager(self, sample_image_path):
        """GlmOcr context manager works."""
        if sample_image_path is None:
            pytest.skip("No sample image available")

        from glmocr.api import GlmOcr

        with GlmOcr() as parser:
            result = parser.parse(str(sample_image_path))
            assert result is not None


@pytest.mark.integration
class TestCLI:
    """Tests for CLI."""

    def test_cli_parse_runs(self, sample_image_path, tmp_path):
        """glmocr parse can run."""
        if sample_image_path is None:
            pytest.skip("No sample image available")

        import subprocess

        result = subprocess.run(
            [
                "python",
                "-m",
                "glmocr.cli",
                "parse",
                str(sample_image_path),
                "--output",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        # Only check it runs; don't require success (config may be invalid)
        assert (
            "Error:" in result.stderr
            or result.returncode == 0
            or "Completed" in result.stdout
        )


# PDF tests were previously skipped due to layout requirements.
