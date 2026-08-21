"""Regression checks for the local dashboard launcher."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cockpit" / "run_local_dashboard.sh"


def test_local_dashboard_launcher_is_valid_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(LAUNCHER)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_local_dashboard_uses_one_configured_python_runtime() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert 'PYTHON_BIN="${SHARPEDGE_PYTHON:-python3}"' in source
    assert "configure_python_runtime" in source
    assert source.index("configure_python_runtime\nstart_server") > source.index(
        "configure_python_runtime()"
    )
    assert '"$PYTHON_BIN" make_cockpit.py' in source
    assert '"$PYTHON_BIN" regime_nerv_panel.py' in source
    assert '"$PYTHON_BIN" make_operator_surface.py' in source
    assert '"$PYTHON_BIN" scripts/nerv_free_data_adapter.py' in source


def test_termux_native_packages_are_reused_without_rebuilding_numpy() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert '"$PREFIX/bin/python3"' in source
    assert "site.getsitepackages()[0]" in source
    assert (
        'export PYTHONPATH="$termux_site_packages${PYTHONPATH:+:$PYTHONPATH}"' in source
    )
    assert "import numpy, requests" in source
