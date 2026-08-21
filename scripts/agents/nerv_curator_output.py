"""Serialization and text rendering for NERV curator packets."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any


def packet_record(packet: Any) -> dict[str, Any]:
    record = asdict(packet)
    record["focus_contracts"] = [asdict(item) for item in packet.focus_contracts]
    return record


def render_packet_text(packet: Any, money: Callable[[Any], str]) -> str:
    lines = [
        "# NERV Curator",
        "",
        packet.headline,
        "",
        f"Stance: {packet.stance}",
        "",
        "## Hey guy summary",
        "",
        packet.hey_guy_summary.get("plain_english", ""),
        "",
        "## Watch next",
        *[f"- {item}" for item in packet.watch_next],
        "",
        "## Focus contracts",
    ]
    for item in packet.focus_contracts:
        lines.append(
            f"- {item.role}: {item.expiration} {item.option_type.upper()} {item.strike:g} "
            f"mid {money(item.mid)} OI {item.open_interest} vol {item.volume} — {item.reason}"
        )
    lines.extend(
        [
            "",
            "## Ignore / de-prioritize",
            *[f"- {item}" for item in packet.noise_filters],
            "",
            "## Warnings",
            *[f"- {item}" for item in packet.warnings],
            "",
        ]
    )
    return "\n".join(lines)


def write_packet_files(
    packet: Any,
    json_path: Path,
    txt_path: Path,
    *,
    render_text: Callable[[Any], str],
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(packet_record(packet), indent=2) + "\n",
        encoding="utf-8",
    )
    txt_path.write_text(render_text(packet), encoding="utf-8")
