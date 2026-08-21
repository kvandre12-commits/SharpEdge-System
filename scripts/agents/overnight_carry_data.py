from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path("data/market_data.db")
DEFAULT_CONTEXT_DB_PATH = Path("data/spy_truth.db")
_CONTEXT_FILTER_COLUMNS = {
    "event_type": "TEXT",
    "open_regime_label": "TEXT",
    "liquidity_regime_type": "TEXT",
    "setup_dir": "TEXT",
    "key_source": "TEXT",
    "vol_state": "TEXT",
    "vol_trend_state": "TEXT",
    "dp_state": "TEXT",
    "macro_state": "TEXT",
    "regime_label": "TEXT",
    "failed_breakdown_open": "INT",
    "accepted_breakdown_open": "INT",
}



def _iso_date(value: Any) -> str:
    return str(value)[:10]



def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None



def _history_cutoff(years: int) -> str:
    cutoff = datetime.now(UTC).date() - timedelta(days=max(years, 1) * 366)
    return cutoff.isoformat()



def _gap_rows_from_price_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gap_rows: list[dict[str, Any]] = []
    prior_close: float | None = None
    for row in rows:
        open_price = _safe_float(row.get("open"))
        session_date = _iso_date(row.get("date"))
        if prior_close and prior_close > 0 and open_price is not None:
            gap_pct = (open_price - prior_close) / prior_close
            gap_rows.append(
                {
                    "session_date": session_date,
                    "prior_close": prior_close,
                    "session_open": open_price,
                    "gap_pct": gap_pct,
                }
            )
        close_price = _safe_float(row.get("close"))
        if close_price is not None and close_price > 0:
            prior_close = close_price
    return gap_rows



def load_gap_history_from_db(
    symbol: str,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    years: int = 10,
) -> dict[str, Any]:
    path = Path(db_path)
    if not path.exists():
        return {
            "available": False,
            "source": str(path),
            "reason": "db_missing",
            "rows": [],
        }
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT date, open, close
            FROM bars_daily
            WHERE symbol = ? AND date >= ?
            ORDER BY date ASC
            """,
            (symbol.upper(), _history_cutoff(years)),
        ).fetchall()
    finally:
        conn.close()
    normalized = [dict(row) for row in rows]
    gap_rows = _gap_rows_from_price_rows(normalized)
    return {
        "available": bool(gap_rows),
        "source": str(path),
        "reason": None if gap_rows else "symbol_missing_or_empty",
        "rows": gap_rows,
    }



def load_gap_history_from_yfinance(symbol: str, *, years: int = 10) -> dict[str, Any]:
    try:
        import yfinance as yf
    except ImportError as exc:
        return {
            "available": False,
            "source": "yfinance",
            "reason": f"import_error:{exc}",
            "rows": [],
        }
    try:
        history = yf.Ticker(symbol.upper()).history(
            period=f"{max(years + 1, 2)}y",
            interval="1d",
            auto_adjust=False,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "source": "yfinance",
            "reason": f"history_error:{exc}",
            "rows": [],
        }
    if history is None or history.empty:
        return {
            "available": False,
            "source": "yfinance",
            "reason": "empty_history",
            "rows": [],
        }
    rows = []
    for index, row in history.iterrows():
        rows.append(
            {
                "date": getattr(index, "date", lambda: index)().isoformat(),
                "open": _safe_float(row.get("Open")),
                "close": _safe_float(row.get("Close")),
            }
        )
    gap_rows = _gap_rows_from_price_rows(rows)
    return {
        "available": bool(gap_rows),
        "source": "yfinance",
        "reason": None if gap_rows else "empty_gap_rows",
        "rows": gap_rows,
    }



def load_gap_history(
    symbol: str,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    years: int = 10,
    allow_yfinance_fallback: bool = True,
) -> dict[str, Any]:
    db_result = load_gap_history_from_db(symbol, db_path=db_path, years=years)
    if db_result.get("available"):
        return db_result
    if not allow_yfinance_fallback:
        return db_result
    yfinance_result = load_gap_history_from_yfinance(symbol, years=years)
    if yfinance_result.get("available"):
        return yfinance_result
    return {
        "available": False,
        "source": f"{db_result.get('source')}|yfinance",
        "reason": yfinance_result.get("reason") or db_result.get("reason") or "unavailable",
        "rows": [],
    }



def _normalized_context_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in (filters or {}).items():
        if key not in _CONTEXT_FILTER_COLUMNS or value in (None, ""):
            continue
        normalized[key] = int(value) if _CONTEXT_FILTER_COLUMNS[key] == "INT" else str(value)
    return normalized



def load_regime_context(
    *,
    proxy_symbol: str,
    filters: dict[str, Any] | None = None,
    db_path: str | Path = DEFAULT_CONTEXT_DB_PATH,
    years: int = 10,
) -> dict[str, Any]:
    normalized_filters = _normalized_context_filters(filters)
    if not normalized_filters:
        return {
            "available": False,
            "source": str(db_path),
            "reason": "no_filters",
            "proxy_symbol": proxy_symbol.upper(),
            "filters": {},
            "session_dates": [],
            "match_count": 0,
        }
    path = Path(db_path)
    if not path.exists():
        return {
            "available": False,
            "source": str(path),
            "reason": "db_missing",
            "proxy_symbol": proxy_symbol.upper(),
            "filters": normalized_filters,
            "session_dates": [],
            "match_count": 0,
        }
    clauses = ["symbol = ?", "session_date >= ?"]
    params: list[Any] = [proxy_symbol.upper(), _history_cutoff(years)]
    for key, value in normalized_filters.items():
        clauses.append(f"{key} = ?")
        params.append(value)
    query = (
        "SELECT session_date FROM auction_expectancy_events WHERE "
        + " AND ".join(clauses)
        + " ORDER BY session_date ASC"
    )
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(query, params).fetchall()
    except sqlite3.Error as exc:
        conn.close()
        return {
            "available": False,
            "source": str(path),
            "reason": f"query_error:{exc}",
            "proxy_symbol": proxy_symbol.upper(),
            "filters": normalized_filters,
            "session_dates": [],
            "match_count": 0,
        }
    finally:
        conn.close()
    session_dates = [_iso_date(row[0]) for row in rows]
    return {
        "available": bool(session_dates),
        "source": str(path),
        "reason": None if session_dates else "no_matching_context_rows",
        "proxy_symbol": proxy_symbol.upper(),
        "filters": normalized_filters,
        "session_dates": session_dates,
        "match_count": len(session_dates),
    }


__all__ = ["load_gap_history", "load_regime_context"]
