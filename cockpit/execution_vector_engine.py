from __future__ import annotations
from datetime import datetime

from bucket_conditioned_spine import build_bucket_conditioned_spine
from day_bucket import classify_day_bucket
from execution_expansion_potential import build_expansion_fuel_surface
from execution_state_scores import (
    score_acceptance_state,
    score_dealer_state,
    score_location_state,
    score_structure_state,
    score_time_state,
    score_trend_state,
)
import execution_vector_context as ctx
import execution_vector_primitives as prim
from execution_card_builder import build_trade_permission_card
from failed_break_facts import failed_break_facts_for_levels
from acceptance_state_engine import build_acceptance_state
from dealer_state_engine import build_dealer_state
from location_state_engine import build_location_state
from time_state_engine import build_time_state
from trend_state_engine import build_trend_state
from volume_profile import build_volume_profile
from vwap_posture import build_vwap_posture
from execution_vector_weights import DEFAULT_BASE_BIAS_WEIGHTS, DEFAULT_BASE_WEIGHTS
from structure_state_engine import build_structure_state
from session_doctrine import minutes_since_open, opening_auction_decay_profile
from range_posture import build_range_posture
from trade_permission_context import (
    BEARISH,
    BULLISH,
    NEUTRAL,
    ScorePart,
    _safe_pct,
    score_balance_context,
    score_compression,
    score_opening_auction,
)


class ExecutionVectorEngine:
    def __init__(self):
        self.base_weights = dict(DEFAULT_BASE_WEIGHTS)
        self.base_bias_weights = dict(DEFAULT_BASE_BIAS_WEIGHTS)
        self.bars, self.pa, self.levels, self.setups = [], {}, {}, []
        self.op, self.gp, self.magnitude, self.full_levels = {}, {}, {}, {}
        self.acceptance_levels = {}
        self.location_references = {}
        self.volatility_structure = None
        self.structure_state = {}
        self.acceptance_state = {}
        self.location_state = {}
        self.dealer_state = {}
        self.volume_state = {}
        self.trend_state = {}
        self.time_state = {}

    def _get_minutes_since_open(self, current_time: datetime) -> float:
        return minutes_since_open(current_time)

    def _score_structure(self):
        structure = self.structure_state or build_structure_state(self.bars)
        return score_structure_state(structure)

    def _score_acceptance(self):
        acceptance = self.acceptance_state or build_acceptance_state(
            self.bars,
            self.acceptance_levels,
        )
        return score_acceptance_state(acceptance)

    def _score_rejection(self):
        """Return last-candle rejection corroboration, not setup identity.

        This is a present-tense microstructure read from the latest candle only.
        It may agree or disagree with setup cards without redefining them.
        """
        if not self.bars:
            return ScorePart(35, NEUTRAL, "no bar data for rejection")
        body, upper_wick, lower_wick, close_pos = prim.bar_personality(self.bars[-1])
        if lower_wick > body * 2 and close_pos > 0.6:
            return ScorePart(70, BULLISH, "last candle rejected lower prices")
        if upper_wick > body * 2 and close_pos < 0.4:
            return ScorePart(70, BEARISH, "last candle rejected higher prices")
        return ScorePart(35, NEUTRAL, "no obvious rejection/trap")

    def _score_trend(self):
        trend = self.trend_state or build_trend_state(self.bars, self.pa)
        return score_trend_state(trend)

    def _score_volume(self):
        profile = self.volume_state or build_volume_profile(self.bars)
        confirmation = str(profile.get("confirmation") or "missing").lower()
        direction = str(profile.get("move_direction") or "flat").lower()
        bias = (
            BULLISH
            if direction == "up" and confirmation in {"confirmed", "participating"}
            else BEARISH
            if direction == "down" and confirmation in {"confirmed", "participating"}
            else NEUTRAL
        )
        reason = str(profile.get("reason") or "no volume profile available")
        if confirmation == "confirmed":
            return ScorePart(85, bias, f"participation confirms move: {reason}")
        if confirmation == "participating":
            return ScorePart(64, bias, f"participation is present: {reason}")
        if confirmation == "mixed":
            return ScorePart(
                42, NEUTRAL, f"participation is mixed behind move: {reason}"
            )
        return ScorePart(25, NEUTRAL, f"participation missing: {reason}")

    def _score_location(self):
        location = self.location_state or build_location_state(
            self.pa.get("spot"),
            self.location_references,
        )
        return score_location_state(location)

    def _score_pressure(self):
        if len(self.bars) < 4:
            return ScorePart(35, NEUTRAL, "need more bars for order-flow pressure")
        vol_mult = self.pa.get("vol_mult", 0)
        recent = self.bars[-4:]
        closes = [bar[4] for bar in recent]
        highs = [bar[2] for bar in recent]
        lows = [bar[3] for bar in recent]
        _body, _upper_wick, _lower_wick, close_pos = prim.bar_personality(recent[-1])
        net = closes[-1] - closes[0]
        up_closes = sum(closes[idx] > closes[idx - 1] for idx in range(1, len(closes)))
        down_closes = sum(
            closes[idx] < closes[idx - 1] for idx in range(1, len(closes))
        )
        displacement = abs(net) / max(max(highs) - min(lows), 1e-9)
        if (
            down_closes >= 3
            and close_pos < 0.30
            and vol_mult >= 1.0
            and displacement >= 0.45
        ):
            return ScorePart(64, BEARISH, "selling pressure persists across bar closes")
        if (
            up_closes >= 3
            and close_pos > 0.70
            and vol_mult >= 1.0
            and displacement >= 0.45
        ):
            return ScorePart(64, BULLISH, "buying pressure persists across bar closes")
        if displacement < 0.25 or (up_closes and down_closes):
            return ScorePart(40, NEUTRAL, "pressure mixed; closes are not one-sided")
        bias = BULLISH if net > 0 else BEARISH if net < 0 else NEUTRAL
        return ScorePart(
            48, bias, "directional pressure exists, but it is not one-sided"
        )

    def _score_time_of_day(self):
        time_state = self.time_state or build_time_state(self.bars)
        return score_time_state(time_state)

    def _score_volatility(self):
        atm_iv = self.op.get("atm_iv") or 0
        premium = self.magnitude.get("premium_read")
        if not atm_iv:
            return ScorePart(50, NEUTRAL, "no volatility read")
        if atm_iv < 0.12:
            base = 52
            reason = "low IV: favor acceptance less, mean reversion more"
        elif atm_iv <= 0.28:
            base = 70
            reason = "normal/high-enough IV for intraday follow-through"
        else:
            base = 60
            reason = "very high IV: moves work, but slippage/whipsaw risk rises"
        if premium == "cheap":
            base += 8
            reason += "; realized move looks cheap vs implied"
        elif premium == "rich":
            base -= 5
            reason += "; options look rich vs realized move"
        return ScorePart(prim.clamp(base), NEUTRAL, reason)

    def _score_exhaustion(self):
        if not self.bars:
            return ScorePart(45, NEUTRAL, "no bars for exhaustion read")
        closes = [bar[4] for bar in self.bars]
        spot = self.pa.get("spot") or closes[-1]
        vwap = self.pa.get("vwap") or spot
        ema20 = ctx.ema(closes[-20:], 20)
        range_posture = build_range_posture(self.pa)
        dist_vwap = abs(spot - vwap) / spot * 100 if spot else 0
        dist_ema = abs(spot - ema20) / spot * 100 if spot and ema20 else 0
        orh = self.full_levels.get("ORH")
        orl = self.full_levels.get("ORL")
        orb_dist = min(
            [abs(spot - level) / spot * 100 for level in (orh, orl) if level],
            default=0.0,
        )
        stretched = (
            bool(range_posture.get("is_stretched_from_value")) or dist_ema >= 0.25
        )
        extreme = bool(range_posture.get("is_extreme"))
        body, upper_wick, lower_wick, _close_pos = prim.bar_personality(self.bars[-1])
        wick_reject = upper_wick > body * 1.8 or lower_wick > body * 1.8
        score = 35 + min(dist_vwap * 55, 22) + min(dist_ema * 70, 20)
        score += 12 if extreme else 0
        score += 10 if wick_reject else 0
        score -= 10 if orb_dist <= 0.12 else 0
        if stretched and str(range_posture.get("side")) == "upside" and extreme:
            return ScorePart(
                prim.clamp(score),
                BEARISH,
                f"stretched high: VWAP {dist_vwap:.2f}%, EMA20 {dist_ema:.2f}%",
            )
        if stretched and str(range_posture.get("side")) == "downside" and extreme:
            return ScorePart(
                prim.clamp(score),
                BULLISH,
                f"stretched low: VWAP {dist_vwap:.2f}%, EMA20 {dist_ema:.2f}%",
            )
        return ScorePart(
            prim.clamp(score),
            NEUTRAL,
            f"not exhausted: VWAP {dist_vwap:.2f}%, EMA20 {dist_ema:.2f}%",
        )

    def _score_trap(self):
        """Return present-tense failed-break trap corroboration.

        Contract doctrine:
        - this is independent corroboration from recent bars + levels only.
        - it is intentionally separate from canonical setup-event identity in
          `detect_failed_breaks()`.
        - disagreement with setup cards is allowed and should be test-covered.
        """
        if not self.bars:
            return ScorePart(35, NEUTRAL, "no bars for trap read")
        level_order = tuple(
            name
            for name in self.full_levels
            if name in {"ORH", "ORL", "PDH", "PDL"}
            and self.full_levels.get(name) is not None
        )
        facts_by_level = failed_break_facts_for_levels(
            self.bars,
            self.full_levels,
            level_names=level_order,
            recent_window=6,
        )
        for name, facts in facts_by_level.items():
            level = facts["level_price"]
            if (
                name in {"ORH", "PDH"}
                and facts["recent_breach_above"]
                and facts.get("current_close_below_level")
            ):
                return ScorePart(
                    78, BEARISH, f"buyers trapped above {name} {level:.2f}"
                )
            if (
                name in {"ORL", "PDL"}
                and facts["recent_breach_below"]
                and facts.get("current_close_above_level")
            ):
                return ScorePart(
                    78, BULLISH, f"sellers trapped below {name} {level:.2f}"
                )
        return ScorePart(35, NEUTRAL, "no failed-break trap detected")

    def _score_dealer_gamma(self):
        dealer = self.dealer_state or build_dealer_state(self.pa, self.op, self.gp)
        return score_dealer_state(dealer)

    def _score_regime(self):
        if len(self.bars) < 10:
            return ScorePart(45, NEUTRAL, "need more bars for regime")
        vwap_posture = build_vwap_posture(self.pa, self.bars)
        mom15 = self.pa.get("mom15", 0)
        vol_mult = self.pa.get("vol_mult", 0)
        range_posture = build_range_posture(
            self.pa,
            vwap_posture=vwap_posture,
        )
        closes = [bar[4] for bar in self.bars]
        first_half = closes[: len(closes) // 2]
        second_half = closes[len(closes) // 2 :]
        drift = _safe_pct(second_half[-1] - first_half[0], first_half[0])
        if vwap_posture.get("is_range_like") and abs(mom15) < 0.08:
            return ScorePart(
                38, NEUTRAL, "balance day: VWAP magnet, mean reversion likely"
            )
        if (
            vol_mult >= 1.0
            and abs(drift) >= 0.35
            and (
                bool(range_posture.get("is_upper_half"))
                or bool(range_posture.get("is_lower_half"))
            )
            and (
                vwap_posture.get("has_upside_control")
                or vwap_posture.get("has_downside_control")
            )
        ):
            bias = (
                BULLISH
                if vwap_posture.get("has_upside_control")
                else BEARISH
                if vwap_posture.get("has_downside_control")
                else NEUTRAL
            )
            return ScorePart(
                82, bias, "trend day regime: VWAP control + directional drift"
            )
        if bool(range_posture.get("is_pressing_edge")):
            bias = BULLISH if str(range_posture.get("side")) == "downside" else BEARISH
            return ScorePart(
                58, bias, "range extreme: continuation needs proof, fade risk exists"
            )
        return ScorePart(
            48, NEUTRAL, "unclear regime; do not overpay for mediocre reads"
        )

    def _opening_auction_decay(self, part):
        current_time = ctx.session_datetime(self.bars)
        minutes_open = self._get_minutes_since_open(current_time)
        profile = opening_auction_decay_profile(minutes_open)
        weight = float(profile["weight"])
        label = str(profile["label"])
        if weight >= 1.0:
            return part
        score = 50 + (part.score - 50) * weight
        bias = part.bias if weight >= 0.5 and abs(score - 50) >= 4 else NEUTRAL
        return ScorePart(
            prim.clamp(score), bias, f"{part.reason}; {label} weight {weight:.1f}x"
        )

    @staticmethod
    def _regime_weight_multiplier(parts):
        trend = parts["trend_score"]
        regime = parts["regime_score"]
        if trend.bias == NEUTRAL or regime.bias == NEUTRAL:
            return 1.0
        if trend.bias == regime.bias:
            return 0.6
        return 1.0

    @staticmethod
    def _pressure_weight_multiplier(parts):
        trend = parts["trend_score"]
        pressure = parts["pressure_score"]
        if trend.bias == NEUTRAL or pressure.bias == NEUTRAL:
            return 1.0
        if trend.bias == pressure.bias:
            return 0.5
        return 1.0

    def _score_weight_map(self, parts):
        weights = dict(self.base_weights)
        weights["regime_score"] *= self._regime_weight_multiplier(parts)
        weights["pressure_score"] *= self._pressure_weight_multiplier(parts)
        return weights

    def _bias_weight_map(self, parts):
        weights = dict(self.base_bias_weights)
        weights["regime_score"] *= self._regime_weight_multiplier(parts)
        weights["pressure_score"] *= self._pressure_weight_multiplier(parts)
        return weights

    @staticmethod
    def _label_to_bias(label: str):
        return {"CALLS": BULLISH, "PUTS": BEARISH}.get(str(label).upper(), NEUTRAL)

    @staticmethod
    def _permission_direction(bias_value):
        if bias_value >= 0.20:
            return BULLISH
        if bias_value <= -0.20:
            return BEARISH
        return NEUTRAL

    @staticmethod
    def _score_to_permission_delta(score):
        return score - 50

    def _directional_permission_delta(self, part, permission_direction):
        delta = self._score_to_permission_delta(part.score)
        if permission_direction == NEUTRAL or part.bias == NEUTRAL:
            return delta
        if part.bias == permission_direction:
            return delta
        return -delta

    def _weighted_score(self, parts, bias_value):
        weights = self._score_weight_map(parts)
        permission_direction = self._permission_direction(bias_value)
        acceptance_part = max(
            (
                parts["acceptance_score"],
                parts["rejection_score"],
                parts["trap_score"],
            ),
            key=lambda part: part.score,
        )
        permission_parts = {
            "structure_score": parts["structure_score"],
            "trend_score": parts["trend_score"],
            "acceptance_score": acceptance_part,
            "volume_score": parts["volume_score"],
            "location_score": parts["location_score"],
            "pressure_score": parts["pressure_score"],
            "time_of_day_score": parts["time_of_day_score"],
            "volatility_score": parts["volatility_score"],
            "opening_auction_score": parts["opening_auction_score"],
            "exhaustion_score": parts["exhaustion_score"],
            "dealer_gamma_score": parts["dealer_gamma_score"],
            "regime_score": parts["regime_score"],
            "compression_score": parts["compression_score"],
            "balance_context_score": parts["balance_context_score"],
        }
        weighted_delta = sum(
            self._directional_permission_delta(
                permission_parts[name], permission_direction
            )
            * weight
            for name, weight in weights.items()
        )
        return prim.clamp(50 + weighted_delta)

    def _weighted_bias(self, parts):
        weights = self._bias_weight_map(parts)
        total = 0.0
        weight_total = 0.0
        for name, weight in weights.items():
            part = parts[name]
            if (
                name == "compression_score"
                and part.reason == "no compression/coil read"
            ):
                continue
            total += part.bias * weight * (part.score / 100)
            weight_total += weight
        return total / max(weight_total, 1e-9)

    def build_parts(
        self,
        bars,
        pa,
        levels,
        setups=None,
        op=None,
        gp=None,
        magnitude=None,
        volatility_structure=None,
    ):
        ctx.bind_engine_context(
            self,
            bars,
            pa,
            levels,
            setups=setups,
            op=op,
            gp=gp,
            magnitude=magnitude,
            volatility_structure=volatility_structure,
        )
        self.structure_state = build_structure_state(self.bars)
        self.acceptance_state = build_acceptance_state(
            self.bars,
            self.acceptance_levels,
        )
        self.location_state = build_location_state(
            self.pa.get("spot"),
            self.location_references,
        )
        self.dealer_state = build_dealer_state(self.pa, self.op, self.gp)
        self.volume_state = build_volume_profile(self.bars)
        self.trend_state = build_trend_state(self.bars, self.pa)
        self.time_state = build_time_state(self.bars)
        parts = {
            "structure_score": self._score_structure(),
            "acceptance_score": self._score_acceptance(),
            "rejection_score": self._score_rejection(),
            "trend_score": self._score_trend(),
            "volume_score": self._score_volume(),
            "location_score": self._score_location(),
            "pressure_score": self._score_pressure(),
            "time_of_day_score": self._score_time_of_day(),
            "volatility_score": self._score_volatility(),
            "opening_auction_score": score_opening_auction(self.bars, self.full_levels),
            "exhaustion_score": self._score_exhaustion(),
            "trap_score": self._score_trap(),
            "dealer_gamma_score": self._score_dealer_gamma(),
            "regime_score": self._score_regime(),
            "compression_score": score_compression(self.pa, self.volatility_structure),
            "balance_context_score": score_balance_context(self.pa),
        }
        parts["opening_auction_score"] = self._opening_auction_decay(
            parts["opening_auction_score"]
        )
        expansion_surface = build_expansion_fuel_surface(
            prim.serialize_parts(parts),
            pa=self.pa,
            gp=self.gp,
        )
        parts["expansion_fuel_score"] = ScorePart(
            int(expansion_surface.get("score") or 0),
            self._label_to_bias(str(expansion_surface.get("bias") or "NEUTRAL")),
            str(expansion_surface.get("reason") or ""),
        )
        return parts

    def build_card(
        self,
        bars,
        pa,
        levels,
        setups=None,
        op=None,
        gp=None,
        magnitude=None,
        volatility_structure=None,
    ):
        parts = self.build_parts(
            bars,
            pa,
            levels,
            setups=setups,
            op=op,
            gp=gp,
            magnitude=magnitude,
            volatility_structure=volatility_structure,
        )
        bias_value = self._weighted_bias(parts)
        raw_permission = self._weighted_score(parts, bias_value)
        market_day = classify_day_bucket(
            parts,
            self.pa,
            self.op,
            self.gp,
            self.setups,
        )
        spine = build_bucket_conditioned_spine(parts, market_day)
        return build_trade_permission_card(
            parts=parts,
            setups=self.setups,
            pa=self.pa,
            raw_permission=raw_permission,
            permission=spine["score"],
            bias_value=bias_value,
            grammar={"authority_engine": "legacy", "mode": "full_stack"},
            market_day=market_day,
            bucket_conditioned_spine=spine,
            score_weights=self._score_weight_map(parts),
            op=self.op,
            gp=self.gp,
            corroboration_parts=parts,
            structure_state=self.structure_state,
            acceptance_state=self.acceptance_state,
            location_state=self.location_state,
            dealer_state=self.dealer_state,
            volume_state=self.volume_state,
            trend_state=self.trend_state,
            time_state=self.time_state,
        )


__all__ = ["ExecutionVectorEngine"]
