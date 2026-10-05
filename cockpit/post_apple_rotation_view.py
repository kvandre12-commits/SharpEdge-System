from __future__ import annotations

import html
from typing import Any

FG = "#e6edf3"
MUTE = "#7d8590"
SURFACE = "#161b22"
GREEN = "#26a641"
RED = "#f85149"
AMBER = "#d29922"
BLUE = "#58a6ff"
WRAP = "overflow-wrap:anywhere;word-break:break-word"


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def _fmt_pts(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}pp"


def _fmt_price(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def _fmt_option_age(seconds: int | None) -> str:
    if seconds is None:
        return "n/a"
    minutes = max(int(seconds) // 60, 0)
    return f"{minutes}m"


def _fmt_option_label(option: dict[str, Any]) -> str:
    strike = option.get("strike")
    option_type = str(option.get("option_type") or "").upper()[:1]
    expiration = option.get("expiration") or "?"
    strike_text = "n/a" if strike is None else f"{float(strike):g}"
    return f"{expiration} {strike_text}{option_type}"


def render_post_apple_rotation_block(packet: dict[str, Any] | None) -> str:
    ctx = packet or {}
    if not ctx.get("available"):
        return ""

    mode = str(ctx.get("mode") or "inactive_window")
    color_map = {
        "trade_today": GREEN,
        "stand_down_context_only": AMBER,
        "inactive_window": BLUE,
    }
    badge_map = {
        "trade_today": "TRADE TODAY",
        "stand_down_context_only": "STAND DOWN",
        "inactive_window": "WINDOW INACTIVE",
    }
    accent = color_map.get(mode, BLUE)
    badge = badge_map.get(mode, mode.replace("_", " ").upper())
    verified = ctx.get("verified_window") or {}
    trades = ctx.get("today_trades") or []

    window_line = (
        f"earnings {_esc(verified.get('earnings_date'))} -> reaction {_esc(verified.get('reaction_session_date'))} "
        f"| current {_esc(verified.get('current_session_date'))} "
        f"| sessions since reaction {_esc(verified.get('sessions_since_reaction'))}"
        if verified.get("available")
        else _esc(verified.get("reason") or "verified window unavailable")
    )
    reaction_line = (
        f"reaction gap {_fmt_pct(verified.get('reaction_open_gap_pct'))} | reaction close vs prior close {_fmt_pct(verified.get('reaction_close_vs_prior_close_pct'))}"
        if verified.get("available")
        else ""
    )

    trade_rows = []
    for row in trades[:3]:
        option = row.get("options_liquidity") or {}
        option_block = ""
        if option.get("available"):
            option_block = (
                f'<div style="margin-top:8px;padding:8px;border:1px solid #30363d;border-radius:6px;background:#11161c">'
                f'<div style="color:{MUTE};font-size:10px;letter-spacing:.08em;font-weight:bold">FLOW POCKET</div>'
                f'<div style="color:{FG};font-size:12px;margin-top:4px">{_esc(_fmt_option_label(option))} '
                f"| bid {_esc(_fmt_price(option.get('bid')))} / ask {_esc(_fmt_price(option.get('ask')))} "
                f"| width {_esc(_fmt_pct(option.get('width_pct')))} </div>"
                f'<div style="color:#adbac7;font-size:11px;margin-top:4px;{WRAP}">'
                f"vol {_esc(_fmt_int(option.get('volume')))} | OI {_esc(_fmt_int(option.get('open_interest')))} "
                f"| age {_esc(_fmt_option_age(option.get('quote_age_seconds')))}"
                f"{' | broker confirm required' if option.get('fresh_quote_required') else ''}</div>"
                "</div>"
            )
        trade_rows.append(
            f'<div style="border:1px solid #30363d;border-radius:8px;padding:10px;margin-top:8px;background:#0f141a">'
            f'<div style="display:flex;justify-content:space-between;gap:8px;align-items:baseline">'
            f'<div><span style="color:{FG};font-size:16px;font-weight:bold">#{int(row.get("rank") or 0)} {_esc(row.get("symbol") or "?")}</span></div>'
            f'<div style="color:{accent};font-size:11px;font-weight:bold">{_esc(row.get("lane_label") or "TRADE")}</div></div>'
            f'<div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;margin-top:8px;color:#adbac7;font-size:12px">'
            f'<div>live day <span style="color:{FG}">{_fmt_pct(row.get("live_day_change_pct"))}</span></div>'
            f'<div>vs QQQ <span style="color:{FG}">{_fmt_pts(row.get("relative_to_qqq_pct_points"))}</span></div>'
            f'<div>vs SPY <span style="color:{FG}">{_fmt_pts(row.get("relative_to_spy_pct_points"))}</span></div>'
            f'<div>range pos <span style="color:{FG}">{_fmt_pct(row.get("live_range_position_pct"))}</span></div>'
            "</div>"
            f'<div style="color:#adbac7;font-size:12px;margin-top:8px;{WRAP}">{_esc(row.get("reason") or "")}</div>'
            f"{option_block}"
            "</div>"
        )

    benchmark_chips = " ".join(
        f'<span style="display:inline-block;padding:2px 8px;border:1px solid #30363d;border-radius:999px;color:#adbac7;font-size:11px;margin:2px 6px 2px 0">'
        f"{_esc(row.get('symbol') or '?')}: {_fmt_pct(row.get('live_day_change_pct'))} vs QQQ {_fmt_pts(row.get('relative_to_qqq_pct_points'))}</span>"
        for row in (ctx.get("benchmark_context") or [])[:2]
    )

    option_source = ctx.get("options_liquidity_source") or {}
    option_source_line = ""
    if option_source.get("available"):
        option_source_line = (
            f'<div style="color:{MUTE};font-size:11px;margin-top:4px;{WRAP}">'
            f"flow pocket snapshot {_esc(option_source.get('updated_at_utc') or 'latest available board')}</div>"
        )

    return (
        f'<div style="border:2px solid {accent};background:{SURFACE};padding:12px;margin:10px 0;border-radius:8px">'
        f'<div style="display:flex;justify-content:space-between;gap:8px;align-items:baseline">'
        f'<div style="color:{MUTE};font-size:11px;letter-spacing:.08em;font-weight:bold">POST-AAPL ROTATION CARD</div>'
        f'<div style="color:{accent};font-size:11px;font-weight:bold">{badge}</div></div>'
        f'<div style="color:{accent};font-size:20px;font-weight:bold;margin-top:4px">{_esc(ctx.get("headline") or "")}</div>'
        f'<div style="color:#adbac7;font-size:13px;margin-top:6px;{WRAP}">{_esc(ctx.get("story") or "")}</div>'
        f'<div style="color:{FG};font-size:12px;margin-top:8px;{WRAP}">{window_line}</div>'
        f'<div style="color:{MUTE};font-size:11px;margin-top:4px;{WRAP}">{reaction_line}</div>'
        f'<div style="color:{MUTE};font-size:11px;margin-top:4px;{WRAP}">{_esc(verified.get("reason") or "")}</div>'
        f'<div style="color:{MUTE};font-size:11px;margin-top:4px;{WRAP}">{_esc(ctx.get("ranking_method") or "")}</div>'
        f"{option_source_line}"
        f'<div style="margin-top:8px">{benchmark_chips}</div>'
        f"{''.join(trade_rows)}"
        "</div>"
    )
