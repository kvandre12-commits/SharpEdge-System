"""Provider-readiness contract tests for NERV."""

from __future__ import annotations

from unittest.mock import patch

from scripts.nerv import provider_status


def test_yfinance_status_distinguishes_configuration_from_availability() -> None:
    def find_spec(module: str):
        return object() if module == "yfinance" else None

    with patch.object(
        provider_status.importlib.util, "find_spec", side_effect=find_spec
    ):
        status = provider_status.yfinance_status().to_record()

    assert status["configured"] is True
    assert status["available"] is False
    assert status["status"] == "dependency_missing"
    assert status["blockers"] == ["dependency_missing:pandas"]


def test_yfinance_status_is_ready_only_when_required_modules_exist() -> None:
    with patch.object(
        provider_status.importlib.util,
        "find_spec",
        return_value=object(),
    ):
        status = provider_status.yfinance_status().to_record()

    assert status["configured"] is True
    assert status["available"] is True
    assert status["status"] == "ready"
    assert status["blockers"] == []


def test_credentialed_provider_reports_explicit_blocker(monkeypatch) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)

    status = provider_status.alpaca_status().to_record()

    assert status["configured"] is False
    assert status["available"] is False
    assert status["status"] == "credentials_missing"
    assert status["blockers"] == ["credentials_missing"]
