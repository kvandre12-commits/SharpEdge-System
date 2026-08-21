"""Render regime/NERV desk boards as a cockpit sidecar panel."""

from __future__ import annotations

import html
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from regime_nerv_tabs import render_tabs_html

ROOT = Path(__file__).resolve().parents[1]
COCKPIT_DIR = Path(__file__).resolve().parent
PANEL_PATH = COCKPIT_DIR / "regime_nerv_panel.html"
SPLIT_PATH = COCKPIT_DIR / "regime_nerv_split.html"
TABS_PATH = COCKPIT_DIR / "regime_nerv_tabs.html"
HEY_GUY_PATH = COCKPIT_DIR / "hey_guy.html"
DEFAULT_REFRESH_SECONDS = 10


@dataclass(frozen=True)
class BoardSource:
    label: str
    path: Path


def discover_board_sources(root: Path = ROOT) -> list[BoardSource]:
    """Return known desk boards, newest first, without requiring generated files."""

    sources = [
        BoardSource("NERV curator", root / "outputs/nerv_curator.json"),
        BoardSource(
            "Standard NERV: SPY/WMT",
            root / "outputs/nerv_cockpit_standard/nerv_liquidity_board.json",
        ),
        BoardSource("NERV: SPY", root / "outputs/nerv_spy/nerv_liquidity_board.json"),
        BoardSource(
            "NERV: SPY month chain",
            root / "outputs/nerv_spy_month/nerv_liquidity_board.json",
        ),
        BoardSource("NERV: WMT", root / "outputs/nerv_wmt/nerv_liquidity_board.json"),
        BoardSource(
            "IV heat: SPY",
            root / "outputs/iv_heat_harvest/spy_iv_heat_harvest.json",
        ),
        BoardSource(
            "CTC/NERV", root / "outputs/nerv_trade_desk/ctc_nerv_trade_desk.json"
        ),
    ]
    cartridge_root = root / "outputs/regime_cartridges"
    if cartridge_root.exists():
        for path in cartridge_root.glob("*/desk/ctc_nerv_trade_desk.json"):
            sources.append(
                BoardSource(f"Cartridge: {path.parents[1].name.upper()}", path)
            )
    existing = [source for source in sources if source.path.exists()]
    return sorted(
        existing, key=lambda source: source.path.stat().st_mtime, reverse=True
    )


def render_panel_html(
    sources: list[BoardSource] | None = None,
    *,
    refresh_seconds: int = DEFAULT_REFRESH_SECONDS,
) -> str:
    sources = discover_board_sources() if sources is None else sources
    cards = []
    for source in sources:
        cards.append(_board_card(source))
    if not cards:
        cards.append(_empty_card())
    return _page("Regime/NERV Desk", "\n".join(cards), refresh_seconds=refresh_seconds)


def _pressure_meter(summary: dict[str, Any]) -> str:
    put_pct = int(summary.get("put_pressure_pct") or 0)
    call_pct = int(summary.get("call_pressure_pct") or 0)
    dominant_side = summary.get("dominant_side") or "unknown"
    return f"""
<div style=\"margin:10px 0 12px\">
  <div class=\"meta\" style=\"margin-bottom:6px\">Pressure split</div>
  <div style=\"display:flex;height:12px;border-radius:999px;overflow:hidden;background:#0b1220;border:1px solid #22324d\">
    <div style=\"width:{put_pct}%;background:#f85149\"></div>
    <div style=\"width:{call_pct}%;background:#3fb950\"></div>
  </div>
  <div style=\"display:flex;justify-content:space-between;gap:10px;margin-top:6px;font-size:12px\">
    <span style=\"color:#f85149\"><b>PUT</b> {_h(put_pct)}% • {_h(summary.get("put_pressure_score") or 0)}</span>
    <span style=\"color:#9fb4d9\">lead: {_h(dominant_side)}</span>
    <span style=\"color:#3fb950\"><b>CALL</b> {_h(call_pct)}% • {_h(summary.get("call_pressure_score") or 0)}</span>
  </div>
</div>
"""


def render_hey_guy_html(*, refresh_seconds: int = DEFAULT_REFRESH_SECONDS) -> str:
    path = ROOT / "outputs/nerv_curator.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        body = _status_card("Hey guy", f"No NERV curator read yet: {exc}", tone="bad")
        return _page("Hey Guy", body, refresh_seconds=refresh_seconds)
    summary = payload.get("hey_guy_summary") or {}
    status = str(summary.get("status") or "degraded")
    blockers = list(summary.get("blockers") or [])
    status_tone = "warn" if status == "degraded" else ""
    status_detail = ", ".join(str(item) for item in blockers) or "none"
    body = f"""
<section class=\"card {status_tone}\" aria-label=\"Surface data status\">
  <h2>Data status: {_h(status)}</h2>
  <p>Blockers: {_h(status_detail)}</p>
</section>
<section class=\"card curator\">
  <h2>{_h(summary.get("title", "Hey guy"))}</h2>
  <div class=\"meta\">Generated: {_h(payload.get("generated_at_utc", "unknown"))}</div>
  <p><b>{_h(summary.get("one_liner", payload.get("headline", "")))}</b></p>
  <p>{_h(summary.get("plain_english", ""))}</p>
  <div class=\"pills\">
    <span class=\"pill\">stance: {_h(summary.get("stance", payload.get("stance", "unknown")))}</span>
    <span class=\"pill\">target: {_h(payload.get("target_strike", ""))}</span>
    <span class=\"pill\">spot: {_h(payload.get("underlying_price", ""))}</span>
  </div>
  <h3>Liquidity spot</h3>
  <p>{_h(summary.get("liquidity_spot", "No clean liquidity spot yet."))}</p>
  <div class=\"pills\">
    <span class=\"pill\">balance: {_h(summary.get("bias_alignment", "unknown"))}</span>
    <span class=\"pill\">flow: {_h(summary.get("flow_balance", "n/a"))}</span>
  </div>
  {_pressure_meter(summary)}
  <p>{_h(summary.get("quote_quality_context", ""))}</p>
  {_bullet_list("Put side", list(summary.get("put_flow") or []))}
  {_bullet_list("Call side", list(summary.get("call_flow") or []))}
  {_bullet_list("Near-money tape", list(summary.get("near_money_tape") or []))}
  {_bullet_list("Confirms", list(summary.get("confirms") or []))}
  {_bullet_list("Invalidates", list(summary.get("invalidates") or []))}
  <h3>Operator note</h3>
  <p>{_h(summary.get("operator_note", "Research-only."))}</p>
</section>
"""
    return _page("Hey Guy", body, refresh_seconds=refresh_seconds)


def render_split_html(*, refresh_seconds: int = DEFAULT_REFRESH_SECONDS) -> str:
    safe_refresh = max(int(refresh_seconds), 1)
    return f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <meta http-equiv=\"refresh\" content=\"{safe_refresh}\">
  <title>SharpEdge Cockpit + Regime/NERV</title>
  <style>
    html, body {{ margin:0; height:100%; background:#05070b; color:#e8eefc; }}
    .wrap {{ display:grid; grid-template-columns:minmax(0, 2fr) minmax(360px, 1fr); height:100vh; }}
    iframe {{ border:0; width:100%; height:100%; background:#08111f; }}
    .right {{ border-left:1px solid #20304d; }}
    .tabs-link {{ position:fixed; right:10px; top:10px; z-index:10; background:#1f6feb; color:white; padding:7px 10px; border-radius:999px; font:12px system-ui; text-decoration:none; }}
    @media (max-width: 900px) {{ .wrap {{ grid-template-columns:1fr; grid-template-rows:62vh 38vh; }} .right {{ border-left:0; border-top:1px solid #20304d; }} }}
  </style>
</head>
<body>
  <a class=\"tabs-link\" href=\"regime_nerv_tabs.html\">tabs</a>
  <div class=\"wrap\">
    <iframe src=\"cockpit.html\" title=\"SharpEdge cockpit\"></iframe>
    <iframe class=\"right\" src=\"regime_nerv_panel.html\" title=\"Regime NERV panel\"></iframe>
  </div>
</body>
</html>
"""


def write_surfaces(
    *,
    refresh_seconds: int = DEFAULT_REFRESH_SECONDS,
    cockpit_dir: Path = COCKPIT_DIR,
    sources: list[BoardSource] | None = None,
) -> dict[str, str]:
    panel = cockpit_dir / "regime_nerv_panel.html"
    split = cockpit_dir / "regime_nerv_split.html"
    tabs = cockpit_dir / "regime_nerv_tabs.html"
    hey_guy = cockpit_dir / "hey_guy.html"
    panel.write_text(
        render_panel_html(sources, refresh_seconds=refresh_seconds), encoding="utf-8"
    )
    split.write_text(
        render_split_html(refresh_seconds=refresh_seconds), encoding="utf-8"
    )
    tabs.write_text(render_tabs_html(refresh_seconds=refresh_seconds), encoding="utf-8")
    hey_guy.write_text(
        render_hey_guy_html(refresh_seconds=refresh_seconds), encoding="utf-8"
    )
    return {
        "panel": str(panel),
        "split": str(split),
        "tabs": str(tabs),
        "hey_guy": str(hey_guy),
    }


def _board_card(source: BoardSource) -> str:
    try:
        payload = json.loads(source.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _status_card(source.label, f"Could not read board: {exc}", tone="bad")
    if payload.get("schema") == "sharpedge.nerv_curator.v1":
        return _curator_card(source, payload)
    if payload.get("schema") == "sharpedge.iv_heat_harvest.v1":
        return _iv_heat_card(source, payload)
    contracts = list(payload.get("contracts") or [])
    if contracts:
        return _contract_board_card(source, payload, contracts)
    rows = list(payload.get("rows") or [])
    summary = payload.get("summary") or {}
    header = _summary_header(source, payload, summary)
    body = _rows_table(rows[:12]) if rows else "<p>No rows in board.</p>"
    return f'<section class="card">{header}{body}</section>'


def _curator_card(source: BoardSource, payload: dict[str, Any]) -> str:
    focus = list(payload.get("focus_contracts") or [])[:8]
    watch_next = list(payload.get("watch_next") or [])[:4]
    warnings = list(payload.get("warnings") or [])[:3]
    iv = payload.get("iv_heat") or {}
    context = payload.get("cockpit_context") or {}
    header = f"""
<h2>{_h(source.label)}</h2>
<div class=\"meta\">Generated: {_h(payload.get("generated_at_utc", "unknown"))}</div>
<div class=\"meta\">{_h(payload.get("headline", "No curator headline."))}</div>
<div class=\"pills\">
  <span class=\"pill\">stance: {_h(payload.get("stance", "unknown"))}</span>
  <span class=\"pill\">IV/RV13: {_h(_fmt_ratio(iv.get("median_iv_rv13_ratio")))}</span>
  <span class=\"pill\">spot: {_h(context.get("spot", payload.get("underlying_price", "")))}</span>
  <span class=\"pill\">target: {_h(payload.get("target_strike", ""))}</span>
</div>
"""
    return (
        f'<section class="card curator">{header}'
        f"{_bullet_list('Watch next', watch_next)}"
        f"{_curator_contract_table(focus)}"
        f"{_bullet_list('Warnings', warnings)}"
        "</section>"
    )


def _bullet_list(title: str, items: list[Any]) -> str:
    if not items:
        return ""
    bullets = "".join(f"<li>{_h(item)}</li>" for item in items)
    return f"<h3>{_h(title)}</h3><ul>{bullets}</ul>"


def _curator_contract_table(contracts: list[dict[str, Any]]) -> str:
    if not contracts:
        return "<p>No curated focus contracts yet.</p>"
    rows = []
    for item in contracts:
        rows.append(
            "<tr>"
            f"<td><b>{_h(item.get('role', ''))}</b><br><span>{_h(item.get('reason', ''))}</span></td>"
            f"<td>{_h(item.get('expiration', ''))}<br><span>{_h(item.get('contract', ''))}</span></td>"
            f"<td>{_h(str(item.get('option_type', '')).upper())} {_h(item.get('strike', ''))}</td>"
            f"<td>{_h(item.get('mid', ''))}<br><span>{_h(item.get('bid', ''))}/{_h(item.get('ask', ''))}</span></td>"
            f"<td>{_h(item.get('volume', ''))}<br><span>OI {_h(item.get('open_interest', ''))}</span></td>"
            f"<td>{_h(item.get('priority', ''))}<br><span>{_h(_short_flags(str(item.get('flags') or '')))}</span></td>"
            "</tr>"
        )
    return (
        """
<h3>Curated focus</h3>
<table>
<thead><tr><th>Role</th><th>Expiry/Contract</th><th>Strike</th><th>Mid</th><th>Vol/OI</th><th>State</th></tr></thead>
<tbody>
"""
        + "\n".join(rows)
        + "\n</tbody></table>"
    )


def _iv_heat_card(source: BoardSource, payload: dict[str, Any]) -> str:
    reads = list(payload.get("expiry_reads") or [])
    header = f"""
<h2>{_h(source.label)}</h2>
<div class=\"meta\">Generated: {_h(payload.get("generated_at_utc", "unknown"))}</div>
<div class=\"meta\">Underlying {_h(payload.get("underlying_price"))} · target strike {_h(payload.get("target_strike"))}</div>
<div class=\"pills\">
  <span class=\"pill\">heat: {_h(payload.get("overall_heat_label", "unknown"))}</span>
  <span class=\"pill\">IV/RV13: {_h(_fmt_ratio(payload.get("median_iv_rv13_ratio")))}</span>
  <span class=\"pill\">event: {_h(payload.get("nearest_event", "none"))} in {_h(payload.get("days_to_nearest_event", ""))}d</span>
</div>
"""
    return f'<section class="card">{header}{_iv_heat_table(reads[:12])}</section>'


def _iv_heat_table(reads: list[dict[str, Any]]) -> str:
    if not reads:
        return "<p>No IV reads available.</p>"
    rows = []
    for row in reads:
        rows.append(
            "<tr>"
            f"<td><b>{_h(row.get('expiration', ''))}</b><br><span>DTE {_h(row.get('dte_calendar', ''))}</span></td>"
            f"<td>{_h(_fmt_pct(row.get('atm_iv_pct')))}<br><span>ATM {_h(row.get('atm_strike', ''))}</span></td>"
            f"<td>{_h(_fmt_ratio(row.get('iv_rv13_ratio')))}<br><span>RV20 {_h(_fmt_ratio(row.get('iv_rv20_ratio')))}</span></td>"
            f"<td>{_h(row.get('heat_label', ''))}<br><span>{_h(row.get('harvest_window', ''))}</span></td>"
            f"<td>{_h(row.get('call_750_mid', ''))}<br><span>IV {_h(_fmt_pct(row.get('call_750_iv_pct')))}</span></td>"
            "</tr>"
        )
    return (
        """
<table>
<thead><tr><th>Expiry</th><th>ATM IV</th><th>IV/RV13</th><th>Heat</th><th>750C</th></tr></thead>
<tbody>
"""
        + "\n".join(rows)
        + "\n</tbody></table>"
    )


def _contract_board_card(
    source: BoardSource, payload: dict[str, Any], contracts: list[dict[str, Any]]
) -> str:
    header = _contract_summary_header(source, payload, contracts)
    body = _contracts_table(contracts[:12])
    return f'<section class="card">{header}{body}</section>'


def _contract_summary_header(
    source: BoardSource, payload: dict[str, Any], contracts: list[dict[str, Any]]
) -> str:
    underlyings = _counts(contract.get("underlying") for contract in contracts)
    priorities = _counts(
        contract.get("manual_validation_priority") for contract in contracts
    )
    expirations = _counts(contract.get("expiration") for contract in contracts)
    generated_at = payload.get("generated_at") or _latest_value(
        contract.get("fetch_timestamp") for contract in contracts
    )
    pill_bits = " ".join(
        [
            *(_pill_bits("underlying", underlyings, limit=4)),
            *(_pill_bits("priority", priorities, limit=3)),
            *(_pill_bits("exp", expirations, limit=3)),
        ]
    )
    return f"""
<h2>{_h(source.label)}</h2>
<div class=\"meta\">Generated: {_h(generated_at or "unknown")}</div>
<div class=\"meta\">Contracts: {_h(len(contracts))} · Research-only liquidity; broker quote required before execution.</div>
<div class=\"pills\">{pill_bits or '<span class="pill">no summary</span>'}</div>
"""


def _contracts_table(contracts: list[dict[str, Any]]) -> str:
    table_rows = []
    for contract in contracts:
        option_type = str(contract.get("option_type") or "?")[:1].upper()
        bid = contract.get("bid")
        ask = contract.get("ask")
        midpoint = contract.get("midpoint")
        flags = str(contract.get("rejection_flags") or "")
        table_rows.append(
            "<tr>"
            f"<td><b>{_h(contract.get('underlying', ''))}</b><br><span>{_h(contract.get('expiration', ''))}</span></td>"
            f"<td>{_h(option_type)} {_h(contract.get('strike', ''))}<br><span>{_h(contract.get('contract_symbol', ''))}</span></td>"
            f"<td>{_h(contract.get('manual_validation_priority', ''))}<br><span>{_h(_short_flags(flags))}</span></td>"
            f"<td>{_h(contract.get('volume', ''))}<br><span>OI {_h(contract.get('open_interest', ''))}</span></td>"
            f"<td>{_h(bid)} / {_h(ask)}<br><span>mid {_h(midpoint)}</span></td>"
            f"<td>{_h(contract.get('nerv_score', ''))}</td>"
            "</tr>"
        )
    return (
        """
<table>
<thead><tr><th>Sym/Exp</th><th>Contract</th><th>State</th><th>Vol/OI</th><th>Bid/Ask</th><th>NERV</th></tr></thead>
<tbody>
"""
        + "\n".join(table_rows)
        + "\n</tbody></table>"
    )


def _summary_header(
    source: BoardSource, payload: dict[str, Any], summary: dict[str, Any]
) -> str:
    states = summary.get("states") or {}
    state_bits = (
        " ".join(
            f'<span class="pill">{_h(key)}: {_h(value)}</span>'
            for key, value in states.items()
        )
        or '<span class="pill">no states</span>'
    )
    return f"""
<h2>{_h(source.label)}</h2>
<div class=\"meta\">Generated: {_h(payload.get("generated_at", "unknown"))}</div>
<div class=\"meta\">Rows: {_h(summary.get("row_count", len(payload.get("rows") or [])))} · Manual validate: {_h(summary.get("manual_validate_count", 0))}</div>
<div class=\"pills\">{state_bits}</div>
"""


def _rows_table(rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{_h(row.get('rank', ''))}</td>"
            f"<td><b>{_h(row.get('ticker', ''))}</b><br><span>{_h(row.get('company', ''))}</span></td>"
            f"<td>{_h(row.get('desk_state', ''))}</td>"
            f"<td>{_h(row.get('structure_family') or row.get('ctc_structure', ''))}<br>"
            f"<span>{_h(row.get('structure_complexity', ''))}</span></td>"
            f"<td>{_h(row.get('nerv_score', ''))}</td>"
            "</tr>"
        )
    return (
        """
<table>
<thead><tr><th>Rank</th><th>Name</th><th>State</th><th>Structure</th><th>NERV</th></tr></thead>
<tbody>
"""
        + "\n".join(table_rows)
        + "\n</tbody></table>"
    )


def _empty_card() -> str:
    return _status_card(
        "No Regime/NERV board found",
        "Run scripts/regime_nerv_trade_desk.py or scripts/ctc_nerv_trade_desk.py. Generated boards are disposable runtime artifacts, so this panel may be empty after cleanup.",
        tone="warn",
    )


def _status_card(title: str, message: str, *, tone: str) -> str:
    return f'<section class="card {tone}"><h2>{_h(title)}</h2><p>{_h(message)}</p></section>'


def _page(title: str, body: str, *, refresh_seconds: int) -> str:
    safe_refresh = max(int(refresh_seconds), 1)
    return f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <meta http-equiv=\"refresh\" content=\"{safe_refresh}\">
  <title>{_h(title)}</title>
  <style>
    body {{ margin:0; padding:12px; font-family:system-ui, sans-serif; background:#07101e; color:#e8eefc; }}
    h1 {{ margin:0 0 10px; font-size:18px; }}
    h2 {{ margin:0 0 6px; font-size:16px; color:#f8d36f; }}
    .card {{ border:1px solid #213858; background:#0d1729; border-radius:12px; padding:10px; margin-bottom:10px; box-shadow:0 0 0 1px #05070b inset; }}
    .warn {{ border-color:#80682a; }} .bad {{ border-color:#8d2e42; }}
    .meta, span {{ color:#9db0cc; font-size:12px; }}
    .pills {{ margin:8px 0; display:flex; flex-wrap:wrap; gap:5px; }}
    .pill {{ border:1px solid #2c466d; border-radius:999px; padding:3px 7px; background:#111f36; color:#c9d8f3; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; }}
    th, td {{ border-top:1px solid #20304d; padding:6px 4px; vertical-align:top; text-align:left; }}
    th {{ color:#9db0cc; font-weight:600; }}
  </style>
</head>
<body>
  <main>
    <h1>Regime/NERV Desk</h1>
    {body}
  </main>
  <script>
    const SCROLL_KEY = 'sharpedge.nervPanel.scrollY:' + location.pathname;
    window.addEventListener('beforeunload', () => sessionStorage.setItem(SCROLL_KEY, String(window.scrollY)));
    window.addEventListener('load', () => {{
      const saved = Number(sessionStorage.getItem(SCROLL_KEY) || 0);
      if (saved > 0) requestAnimationFrame(() => window.scrollTo(0, saved));
    }});
  </script>
</body>
</html>
"""


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _pill_bits(prefix: str, counts: dict[str, int], *, limit: int) -> list[str]:
    return [
        f'<span class="pill">{_h(prefix)} {_h(key)}: {_h(value)}</span>'
        for key, value in list(counts.items())[:limit]
    ]


def _latest_value(values: Any) -> str:
    clean = sorted(str(value) for value in values if value)
    return clean[-1] if clean else ""


def _short_flags(flags: str) -> str:
    if not flags:
        return "ok"
    parts = [part for part in flags.split(";") if part]
    return ";".join(parts[:3])


def _fmt_pct(value: Any) -> str:
    parsed = _as_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.1f}%"


def _fmt_ratio(value: Any) -> str:
    parsed = _as_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.2f}x"


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _h(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


if __name__ == "__main__":
    refresh = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REFRESH_SECONDS
    paths = write_surfaces(refresh_seconds=refresh)
    print(f"[regime/nerv panel] panel: {paths['panel']}")
    print(f"[regime/nerv panel] split: {paths['split']}")
    print(f"[regime/nerv panel] tabs: {paths['tabs']}")
    print(f"[regime/nerv panel] hey_guy: {paths['hey_guy']}")
