"""Packet assembly for transition pressure."""

from __future__ import annotations

from typing import Any

from .deltas import build_transition_deltas
from .interactions import build_interactions
from .potential_energy import build_potential_energy
from .pressure import build_transition_pressure_state


def build_transition_pressure_packet(
    pa: dict[str, Any],
    op: dict[str, Any],
    gp: dict[str, Any],
    volatility_structure: dict[str, Any],
    setups: list[dict[str, Any]] | None,
    current_receipt: dict[str, Any],
    prior_receipts: list[dict[str, Any]],
    level_states: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    deltas = build_transition_deltas(current_receipt, prior_receipts)
    potential_energy = build_potential_energy(
        pa,
        op,
        gp,
        volatility_structure,
        setups,
        current_receipt,
        level_states=level_states,
    )
    interactions = build_interactions(deltas, potential_energy)
    pressure = build_transition_pressure_state(
        pa,
        deltas,
        potential_energy,
        interactions,
        current_receipt,
        prior_receipts,
    )
    return {
        **pressure,
        "deltas": deltas,
        "potential_energy": potential_energy,
        "interactions": interactions,
    }


__all__ = ["build_transition_pressure_packet"]
