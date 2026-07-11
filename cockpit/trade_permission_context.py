from __future__ import annotations

from dataclasses import dataclass

BULLISH = 1
BEARISH = -1
NEUTRAL = 0


@dataclass(frozen=True)
class ScorePart:
    score: int
    bias: int = NEUTRAL
    reason: str = ""


def _safe_pct(num, den):
    return num / den * 100 if den else 0.0


def _last_close(bars):
    return bars[-1][4] if bars else 0.0


def score_compression(pa, volatility_structure=None):
    state = volatility_structure or {}
    if not state:
        return ScorePart(50, NEUTRAL, "no compression/coil read")
    volatility_state = state.get("volatility_state")
    structure_state = state.get("structure_state")
    directional_bias = (state.get("bias") or "neutral").lower()
    spot = pa.get("spot") or 0.0
    trigger_high = state.get("trigger_high")
    trigger_low = state.get("trigger_low")
    near_high = bool(
        spot and trigger_high and abs(trigger_high - spot) / spot * 100 <= 0.05
    )
    near_low = bool(
        spot and trigger_low and abs(spot - trigger_low) / spot * 100 <= 0.05
    )

    if state.get("coil") and structure_state == "channel_breakout_setup":
        if directional_bias == "neutral_to_bearish":
            if near_low:
                return ScorePart(
                    74, BEARISH, "post-selloff coil pressing breakdown trigger"
                )
            if near_high:
                return ScorePart(
                    46,
                    NEUTRAL,
                    "post-selloff coil at reclaim trigger; acceptance not proven",
                )
            return ScorePart(
                40,
                BEARISH,
                "post-selloff coil mid-channel; breakdown/reclaim direction unresolved",
            )
        if directional_bias == "neutral_to_bullish":
            if near_high:
                return ScorePart(
                    74, BULLISH, "post-selloff coil pressing breakout trigger"
                )
            if near_low:
                return ScorePart(
                    46,
                    NEUTRAL,
                    "post-selloff coil near failure trigger; reclaim not proven",
                )
            return ScorePart(
                40,
                BULLISH,
                "post-selloff coil mid-channel; breakout/failure direction unresolved",
            )
        return ScorePart(
            42, NEUTRAL, "coil detected, but directional trigger is unresolved"
        )

    if volatility_state == "squeeze" and structure_state == "narrow_channel":
        return ScorePart(44, NEUTRAL, "tight squeeze with no displacement edge yet")
    if volatility_state == "contraction":
        return ScorePart(
            52, NEUTRAL, "volatility contracting; expansion setup is forming"
        )
    if volatility_state == "expansion":
        return ScorePart(
            58,
            NEUTRAL,
            "volatility expanding; trend continuation or exhaustion can accelerate",
        )
    return ScorePart(50, NEUTRAL, "compression state is informative but not decisive")


def score_opening_auction(bars, levels):
    pdc = levels.get("PDC")
    if not pdc or not bars:
        return ScorePart(50, NEUTRAL, "no prior close for gap read")
    open_ = bars[0][1]
    close = _last_close(bars)
    gap_pct = _safe_pct(open_ - pdc, pdc)
    if abs(gap_pct) < 0.15:
        return ScorePart(55, NEUTRAL, f"flat open ({gap_pct:+.2f}% gap)")
    if gap_pct > 0 and close > open_:
        return ScorePart(72, BULLISH, f"gap up accepting ({gap_pct:+.2f}%)")
    if gap_pct > 0 and close < pdc:
        return ScorePart(36, BEARISH, f"gap up failed and filled ({gap_pct:+.2f}%)")
    if gap_pct < 0 and close < open_:
        return ScorePart(72, BEARISH, f"gap down accepting ({gap_pct:+.2f}%)")
    if gap_pct < 0 and close > pdc:
        return ScorePart(66, BULLISH, f"gap down reclaimed ({gap_pct:+.2f}%)")
    return ScorePart(48, NEUTRAL, f"gap is unresolved ({gap_pct:+.2f}%)")


def _bias_from_label(label):
    if label == "CALLS":
        return BULLISH
    if label == "PUTS":
        return BEARISH
    return NEUTRAL


def score_balance_context(pa):
    confluence = (pa or {}).get("balance_confluence") or {}
    disagreement = (pa or {}).get("balance_disagreement") or {}
    flip = (pa or {}).get("dominant_balance_flip") or {}
    score = int(confluence.get("score", 50) or 50)
    bias = _bias_from_label(confluence.get("bias"))
    reasons = [confluence.get("reason") or "balance context unavailable"]
    if disagreement.get("has_disagreement"):
        score = min(score, 30)
        bias = NEUTRAL
        reasons = [disagreement.get("reason") or "balance lenses disagree"]
    if flip.get("flipped"):
        score = max(26, score - 8)
        if confluence.get("agreement_count", 0) < 3:
            bias = NEUTRAL
        reasons.append(flip.get("reason") or "dominant balance lens flipped")
    return ScorePart(score, bias, "; ".join(reason for reason in reasons if reason))
