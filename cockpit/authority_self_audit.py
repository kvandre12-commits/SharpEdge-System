"""Self-audit contract for cockpit execution authority semantics.

This module does not score the market. It scores the scorer, because apparently
we are adults now. The cockpit may publish useful execution reads, but broker or
operator authority must remain explicitly separated from authored score buckets.
"""

from __future__ import annotations

from typing import Any

DIRTY_FULL_STACK_GAPS = (
    "legacy full_stack blends core, secondary, context, and suspect drift voices",
    "score thresholds are authored buckets, not empirically calibrated probabilities",
    "historical audit flags leakage in location, acceptance, and dealer/gamma lenses",
)

CORE_SPINE_GAPS = (
    "core spine is cleaner than full_stack but still uses authored score buckets",
    "bucket offsets and 58/72 gates are doctrine, not validated calibration curves",
    "score evidence supports cockpit judgment; it does not authorize broker action",
)

TIGHTENED_CORE_FACTS = (
    "core authority lane excludes trap/rejection secondary confirmations",
    "core authority lane excludes pressure/regime suspect drift voices",
    "core verticals are backed by explicit state packets where available",
)

FINAL_AUTHORITY_SOURCE = "approval_decision_plus_operator"
SCORE_SPINE_ROLE = "diagnostic_advisory"


def build_authority_self_audit(
    *,
    authority_engine: str,
    authority_mode: str,
    bucket_conditioned_spine: dict[str, Any],
    raw_permission: int,
    permission: int,
) -> dict[str, Any]:
    """Return an explicit audit packet for authority/display consumers.

    The current score spine is explainable and useful, but not calibrated tightly
    enough to be labeled final execution authority. Keep the read; demote the
    authority semantics. Boring? Yes. Correct? Also yes.
    """
    engine = str(authority_engine or "legacy")
    mode = str(authority_mode or "full_stack")
    is_core = mode == "core_spine_only" or engine == "ace"
    gaps = list(CORE_SPINE_GAPS if is_core else DIRTY_FULL_STACK_GAPS)
    tightened = list(TIGHTENED_CORE_FACTS if is_core else ())
    status = "demoted_pending_calibration" if is_core else "demoted_dirty_full_stack"
    return {
        "schema": "sharpedge.authority_self_audit.v1",
        "status": status,
        "authority_engine": engine,
        "authority_mode": mode,
        "final_authority_source": FINAL_AUTHORITY_SOURCE,
        "score_spine_role": SCORE_SPINE_ROLE,
        "score_spine_authority": SCORE_SPINE_ROLE,
        "display_headline": "EXECUTION READ",
        "display_note": (
            "Score spine is diagnostic/advisory. Final permission requires "
            "approval_decision plus the operator/human gate."
        ),
        "spine_gate": bucket_conditioned_spine.get("gate"),
        "spine_score": bucket_conditioned_spine.get("score", permission),
        "raw_vector_permission_score": int(raw_permission),
        "published_permission_score": int(permission),
        "tightened_facts": tightened,
        "remaining_gaps": gaps,
        "demotion_reason": "; ".join(gaps[:2]),
    }


__all__ = ["build_authority_self_audit"]
