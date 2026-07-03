from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CapabilityProfile:
    read_state: bool = True
    read_execution_card: bool = True
    read_permission: bool = True
    read_positions: bool = True
    read_handoff: bool = True
    prepare_handoff: bool = True
    validate_handoff: bool = True
    execute_handoff: bool = False

    def as_dict(self) -> dict[str, bool]:
        return asdict(self)

    def allows(self, capability: str) -> bool:
        return bool(getattr(self, capability, False))


def default_capabilities() -> CapabilityProfile:
    return CapabilityProfile()
