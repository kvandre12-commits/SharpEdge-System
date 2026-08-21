"""Evaluator-first contracts for SharpEdge's paper-only alpha swarm."""

from scripts.alpha_swarm.contracts import ContractError, manifest_sha256
from scripts.alpha_swarm.evaluator import evaluate_candidate, score_receipts

__all__ = [
    "ContractError",
    "evaluate_candidate",
    "manifest_sha256",
    "score_receipts",
]
