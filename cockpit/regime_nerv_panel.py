"""Render regime/NERV desk boards as a cockpit sidecar panel."""

from __future__ import annotations

import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
COCKPIT_DIR = Path(__file__).resolve().parent
PANEL_PATH = COCKPIT_DIR / "regime_nerv_panel.html"
SPLIT_PATH = COCKPIT_DIR / "regime_nerv_split.html"
DEFAULT_REFRESH_SECONDS = 10


@dataclass(frozen=True)
class BoardSource:
    label: str
    path: Path


def discover_board_sources(root: Path = ROOT) -> list[BoardSource]:
    """Return known desk boards, newest first, without requiring generated files."""

    sources = [
        BoardSource(
            "CTC/NERV", root / "outputs/nerv_trade_desk/ctc_nerv_trade_desk.json"
        )
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
    @media (max-width: 900px) {{ .wrap {{ grid-template-columns:1fr; grid-template-rows:62vh 38vh; }} .right {{ border-left:0; border-top:1px solid #20304d; }} }}
  </style>
</head>
<body>
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
) -> dict[str, str]:
    panel = cockpit_dir / "regime_nerv_panel.html"
    split = cockpit_dir / "regime_nerv_split.html"
    panel.write_text(
        render_panel_html(refresh_seconds=refresh_seconds), encoding="utf-8"
    )
    split.write_text(
        render_split_html(refresh_seconds=refresh_seconds), encoding="utf-8"
    )
    return {"panel": str(panel), "split": str(split)}


def _board_card(source: BoardSource) -> str:
    try:
        payload = json.loads(source.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _status_card(source.label, f"Could not read board: {exc}", tone="bad")
    rows = list(payload.get("rows") or [])
    summary = payload.get("summary") or {}
    header = _summary_header(source, payload, summary)
    body = _rows_table(rows[:12]) if rows else "<p>No rows in board.</p>"
    return f'<section class="card">{header}{body}</section>'


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
  <h1>Regime/NERV Desk</h1>
  {body}
</body>
</html>
"""


def _h(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


if __name__ == "__main__":
    refresh = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REFRESH_SECONDS
    paths = write_surfaces(refresh_seconds=refresh)
    print(f"[regime/nerv panel] panel: {paths['panel']}")
    print(f"[regime/nerv panel] split: {paths['split']}")
