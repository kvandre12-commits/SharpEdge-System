"""Small deterministic helpers for content-addressed agent receipts."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(payload: dict[str, Any]) -> str:
    """Serialize a JSON object deterministically for hashing and comparison."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def payload_sha256(payload: dict[str, Any]) -> str:
    """Return the SHA-256 digest of a canonical JSON object."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


__all__ = ["canonical_json", "payload_sha256"]
