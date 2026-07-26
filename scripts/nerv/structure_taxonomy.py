"""Strategy taxonomy for regime/NERV desk rows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyClassification:
    family: str
    complexity: str
    manual_complex_review_required: bool
    reason: str


def classify_strategy(
    *,
    structure: str = "",
    preferred_structure: str = "",
    preferred_vehicle: str = "",
) -> StrategyClassification:
    """Classify a proposed structure without pretending it is executable."""

    text = " ".join([structure, preferred_structure, preferred_vehicle]).lower()
    compact = text.replace("-", " ").replace("_", " ")

    if "back ratio" in compact or "backratio" in compact or "ratio back" in compact:
        return _complex("back_ratio", "convex ratio structure with short-leg risk")
    if "ratio diagonal" in compact or ("ratio" in compact and "diagonal" in compact):
        return _complex(
            "ratio_diagonal", "diagonal plus ratio sizing requires payoff/margin review"
        )
    if "ratio" in compact:
        return _complex("ratio_spread", "ratio sizing requires payoff/margin review")
    if "diagonal" in compact:
        return _complex(
            "diagonal", "different expirations require calendar/assignment review"
        )
    if "calendar" in compact:
        return _complex(
            "calendar", "different expirations require calendar/assignment review"
        )
    if "covered" in compact or "income" in compact:
        return _complex(
            "income_overlay",
            "overlay income structures need assignment/dividend review",
        )
    if "leaps" in compact or "long dated" in compact or "long-dated" in text:
        return _complex(
            "long_dated_options",
            "long-dated options need separate duration/risk review",
        )
    if "rework" in compact or "reject" in compact:
        return _pending(
            "needs_rework", "workbook marks structure for replacement or rejection"
        )
    if "branch" in compact and "debit spread" in compact:
        return _pending(
            "branch_defined_debit_spread",
            "directional branch must choose call/put/no-trade",
        )
    if "call" in compact and "debit spread" in compact:
        return _vanilla("call_debit_spread", "defined-risk vertical debit spread")
    if "put" in compact and "debit spread" in compact:
        return _vanilla("put_debit_spread", "defined-risk vertical debit spread")
    if "debit spread" in compact:
        return _vanilla("debit_spread", "defined-risk vertical debit spread")
    if "spread" in compact and "option" in compact:
        return _vanilla(
            "option_spread", "spread-like option structure needs leg confirmation"
        )
    if "share" in compact or "equity" in compact:
        return StrategyClassification(
            family="equity_or_shares",
            complexity="equity_research",
            manual_complex_review_required=False,
            reason="equity/share vehicle is not an options structure",
        )
    return StrategyClassification(
        family="unknown",
        complexity="unspecified",
        manual_complex_review_required=False,
        reason="no concrete structure family detected",
    )


def _vanilla(family: str, reason: str) -> StrategyClassification:
    return StrategyClassification(
        family=family,
        complexity="vanilla_defined_risk",
        manual_complex_review_required=False,
        reason=reason,
    )


def _complex(family: str, reason: str) -> StrategyClassification:
    return StrategyClassification(
        family=family,
        complexity="complex_manual_review",
        manual_complex_review_required=True,
        reason=reason,
    )


def _pending(family: str, reason: str) -> StrategyClassification:
    return StrategyClassification(
        family=family,
        complexity="branch_pending",
        manual_complex_review_required=False,
        reason=reason,
    )
