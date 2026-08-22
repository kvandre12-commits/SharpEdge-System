"""Dead-simple options-walls box for `sharpedge.signal.v1`.

A minimal, phone-friendly view of the levels that matter most intraday: where
price sits between the put wall (support) and call wall (resistance), plus the
gamma magnets (pin, max_pain). Interpretation only — no trade authorization.

`build_walls_html(signal)` is pure (dict -> HTML string), so it renders from
`outputs/signal.json` or any signal-shaped dict and is trivially testable.
"""

from __future__ import annotations

import html
from typing import Any


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: float | None, prefix: str = "", suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{prefix}{value:,.2f}{suffix}"


def _delta_from_spot(level: float | None, spot: float | None) -> str:
    if level is None or spot is None or spot == 0:
        return "—"
    diff = level - spot
    pct = diff / spot * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:,.2f}  ({sign}{pct:.2f}%)"


def _position_pct(spot: float | None, low: float | None, high: float | None) -> float:
    """Where spot sits between put wall (0%) and call wall (100%), clamped."""
    if None in (spot, low, high) or high == low:
        return 50.0
    return max(0.0, min(100.0, (spot - low) / (high - low) * 100.0))


def build_walls_html(signal: dict[str, Any]) -> str:
    """Render the walls box HTML from a signal dict."""
    spot = _num(signal.get("spot"))
    call_wall = _num(signal.get("call_wall"))
    put_wall = _num(signal.get("put_wall"))
    pin = _num(signal.get("pin"))
    max_pain = _num(signal.get("max_pain"))
    pcr = _num(signal.get("pcr"))
    atm_iv = _num(signal.get("atm_iv"))
    day_chg = _num(signal.get("day_chg"))
    regime = str(signal.get("gamma_regime") or "unknown").lower()
    expiry = html.escape(str(signal.get("exp") or signal.get("expiry") or ""))
    ts = html.escape(str(signal.get("ts") or signal.get("timestamp") or ""))

    if regime == "negative":
        regime_pill, regime_class = "NEG \u03b3 \u00b7 RUNNER", "runner"
    elif regime == "positive":
        regime_pill, regime_class = "POS \u03b3 \u00b7 STICKY", "sticky"
    else:
        regime_pill, regime_class = "\u03b3 \u00b7 UNKNOWN", "neutral"

    day_sign = "+" if (day_chg or 0) >= 0 else ""
    day_str = "—" if day_chg is None else f"{day_sign}{day_chg:.2f}%"

    # Levels sorted high -> low; spot highlighted as YOU ARE HERE.
    rows = [
        ("call wall", "resistance", call_wall, "resist"),
        ("max pain", "slow magnet", max_pain, "magnet"),
        ("pin", "gamma magnet", pin, "magnet"),
        ("SPOT", "you are here", spot, "spot"),
        ("put wall", "support", put_wall, "support"),
    ]
    rows = [r for r in rows if r[2] is not None]
    rows.sort(key=lambda r: r[2], reverse=True)

    ladder = "\n".join(
        f'<div class="lvl {cls}">'
        f'<span class="name">{html.escape(name)}</span>'
        f'<span class="px">{_fmt(px)}</span>'
        f'<span class="sub">{html.escape(sub)}</span>'
        f'<span class="delta">{"" if cls == "spot" else _delta_from_spot(px, spot)}</span>'
        f"</div>"
        for name, sub, px, cls in rows
    )

    pos = _position_pct(spot, put_wall, call_wall)
    iv_str = "—" if atm_iv is None else f"{atm_iv * 100:.1f}%"
    stale_note = "markets closed / last session — treat as stale" if not ts else f"as of {ts}"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SharpEdge — Walls</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; background:#0b0e13; color:#e6e9ef;
         font:16px/1.4 -apple-system,Segoe UI,Roboto,sans-serif; padding:16px; }}
  .card {{ max-width:520px; margin:0 auto; }}
  .top {{ display:flex; align-items:baseline; justify-content:space-between; gap:8px; }}
  .spot {{ font-size:34px; font-weight:800; }}
  .chg {{ font-size:16px; color:#8b93a7; }}
  .pill {{ padding:4px 10px; border-radius:999px; font-size:12px; font-weight:700; }}
  .pill.runner {{ background:#3a1620; color:#ff6b81; }}
  .pill.sticky {{ background:#123024; color:#4ade80; }}
  .pill.neutral {{ background:#22262f; color:#9aa3b2; }}
  .gauge {{ height:10px; border-radius:6px; margin:14px 0 4px;
           background:linear-gradient(90deg,#4ade80,#22262f,#ff6b81); position:relative; }}
  .gauge .you {{ position:absolute; top:-4px; width:3px; height:18px; background:#fff;
                 border-radius:2px; transform:translateX(-50%); }}
  .gaugelbl {{ display:flex; justify-content:space-between; font-size:11px; color:#8b93a7; }}
  .ladder {{ margin-top:16px; border:1px solid #22262f; border-radius:12px; overflow:hidden; }}
  .lvl {{ display:grid; grid-template-columns:1fr auto 1fr; grid-auto-rows:auto;
          gap:2px 10px; padding:10px 14px; border-bottom:1px solid #171b22; align-items:center; }}
  .lvl:last-child {{ border-bottom:none; }}
  .lvl .name {{ font-weight:700; }}
  .lvl .px {{ font-variant-numeric:tabular-nums; font-weight:800; font-size:18px; text-align:center; }}
  .lvl .sub {{ font-size:11px; color:#8b93a7; }}
  .lvl .delta {{ grid-column:3; text-align:right; font-variant-numeric:tabular-nums;
                 font-size:12px; color:#8b93a7; }}
  .lvl.resist {{ background:#160f13; }} .lvl.resist .px {{ color:#ff6b81; }}
  .lvl.support {{ background:#0f1512; }} .lvl.support .px {{ color:#4ade80; }}
  .lvl.spot {{ background:#161d2b; }} .lvl.spot .px {{ color:#7cc4ff; }}
  .lvl.magnet .px {{ color:#e8c37a; }}
  .foot {{ display:flex; flex-wrap:wrap; gap:6px 16px; margin-top:14px;
           font-size:13px; color:#9aa3b2; }}
  .foot b {{ color:#e6e9ef; font-weight:700; }}
  .stale {{ margin-top:12px; font-size:12px; color:#c98b4b; }}
</style></head>
<body><div class="card">
  <div class="top">
    <div><span class="spot">SPY {_fmt(spot, "$")}</span> <span class="chg">{day_str}</span></div>
    <span class="pill {regime_class}">{regime_pill}</span>
  </div>
  <div class="gauge"><div class="you" style="left:{pos:.1f}%"></div></div>
  <div class="gaugelbl"><span>put wall {_fmt(put_wall, "$")}</span><span>call wall {_fmt(call_wall, "$")}</span></div>
  <div class="ladder">
{ladder}
  </div>
  <div class="foot">
    <span>PCR <b>{_fmt(pcr)}</b></span>
    <span>ATM IV <b>{iv_str}</b></span>
    <span>max pain <b>{_fmt(max_pain, "$")}</b></span>
    <span>pin <b>{_fmt(pin, "$")}</b></span>
    {f'<span>exp <b>{expiry}</b></span>' if expiry else ''}
  </div>
  <div class="stale">{html.escape(stale_note)}</div>
</div></body></html>"""


__all__ = ["build_walls_html"]
