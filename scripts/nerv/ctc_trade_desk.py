"""CTC/NERV trade-desk board builder.

This module joins three things that should never have been trapped in separate
spreadsheet goblin caves:

- CTC catalyst/disposition workbook rows;
- NERV option-chain liquidity board rows;
- SharpEdge's research-only execution gate.

It produces a desk board for human review. It does not authorize orders.
"""

from __future__ import annotations

import csv
import json
import posixpath
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from .models import RESEARCH_ONLY_WARNING, utc_now_iso
from .structure_taxonomy import StrategyClassification, classify_strategy

XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RELS_NS = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

DEFAULT_CTC_WORKBOOK = Path(
    "/sdcard/Download/CTC_C001_Full_34_Name_Disposition_v0_5 (1).xlsx"
)
DEFAULT_NERV_BOARD = Path("outputs/nerv/nerv_liquidity_board.json")
DEFAULT_OUTPUT_DIR = Path("outputs/nerv_trade_desk")

DESK_CSV_FIELDS = [
    "rank",
    "ticker",
    "company",
    "sleeve",
    "primary_mechanism",
    "current_research_state",
    "preferred_vehicle",
    "preferred_structure",
    "disposition_score",
    "thesis_score",
    "ctc_structure",
    "ctc_expiry",
    "ctc_dte",
    "ctc_net_debit_proxy",
    "structure_family",
    "structure_complexity",
    "structure_taxonomy_reason",
    "manual_complex_review_required",
    "nerv_best_contract",
    "nerv_best_expiration",
    "nerv_best_option_type",
    "nerv_best_strike",
    "nerv_score",
    "nerv_priority",
    "nerv_rejection_flags",
    "desk_state",
    "execute_block_reason",
    "next_action",
]


@dataclass(frozen=True)
class CTCUniverseRow:
    ticker: str
    company: str = ""
    sleeve: str = ""
    primary_mechanism: str = ""
    current_research_state: str = ""
    preferred_vehicle: str = ""
    preferred_structure: str = ""
    disposition_score: float | None = None
    thesis_score: float | None = None
    rank: int | None = None


@dataclass(frozen=True)
class CTCTradeRow:
    ticker: str
    structure: str = ""
    expiry: str = ""
    dte: int | None = None
    net_debit_proxy: float | None = None
    max_loss_proxy: float | None = None


def load_xlsx_tables(path: str | Path) -> dict[str, list[dict[str, str]]]:
    """Return sheet tables from an .xlsx using only stdlib zip/xml.

    We intentionally avoid adding a heavyweight Excel dependency. If Excel can
    save it, this parser can read the normal shared-string worksheet shape.
    Fancy merged-cell vibes are ignored because merged cells are layout, not data.
    """

    workbook_path = Path(path)
    with ZipFile(workbook_path) as archive:
        shared = _read_shared_strings(archive)
        sheet_paths = _workbook_sheet_paths(archive)
        return {
            title: _sheet_table(archive, sheet_path, shared)
            for title, sheet_path in sheet_paths.items()
        }


def build_trade_desk_payload(
    *,
    ctc_workbook: str | Path,
    nerv_board_path: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    tables = load_xlsx_tables(ctc_workbook)
    universe = _load_universe(tables)
    trade_rows = _load_trade_rows(tables)
    execution_checks = _load_execution_checks(tables)
    nerv_by_symbol = _load_nerv_by_symbol(nerv_board_path)

    rows = []
    for ticker, ctc in sorted(universe.items(), key=lambda item: _sort_key(item[1])):
        trade = trade_rows.get(ticker, CTCTradeRow(ticker=ticker))
        execution_check = execution_checks.get(ticker, {})
        nerv = nerv_by_symbol.get(ticker)
        rows.append(_build_desk_row(ctc, trade, execution_check, nerv))

    summary = _summary(rows)
    payload = {
        "schema": "sharpedge.ctc_nerv_trade_desk.v1",
        "generated_at": generated_at or utc_now_iso(),
        "ctc_workbook": str(ctc_workbook),
        "nerv_board_path": str(nerv_board_path),
        "summary": summary,
        "governance": {
            "mode": "research_only_human_in_the_loop",
            "fresh_quote_required": True,
            "broker_execution_gate_required": True,
            "non_spy_order_policy": "requires_expanded_operator_approval_and_risk_limits",
            "warning": RESEARCH_ONLY_WARNING,
        },
        "rows": rows,
    }
    return payload


def write_trade_desk_artifacts(
    payload: dict[str, Any],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "ctc_nerv_trade_desk.json"
    csv_path = out / "ctc_nerv_trade_desk.csv"
    md_path = out / "ctc_nerv_trade_desk.md"

    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=DESK_CSV_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(payload["rows"])
    md_path.write_text(_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}


def _read_shared_strings(archive: ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    strings: list[str] = []
    for item in root.findall("a:si", XLSX_NS):
        strings.append(
            "".join(node.text or "" for node in item.findall(".//a:t", XLSX_NS))
        )
    return strings


def _workbook_sheet_paths(archive: ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall("rel:Relationship", RELS_NS)
    }
    paths: dict[str, str] = {}
    for sheet in workbook.findall("a:sheets/a:sheet", XLSX_NS):
        target = rel_map.get(sheet.attrib.get(RID, ""), "")
        if not target:
            continue
        paths[sheet.attrib.get("name", "Sheet")] = _resolve_xl_target(target)
    return paths


def _resolve_xl_target(target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    if target.startswith("xl/"):
        return target
    return posixpath.normpath("xl/" + target)


def _sheet_table(
    archive: ZipFile, sheet_path: str, shared: list[str]
) -> list[dict[str, str]]:
    root = ET.fromstring(archive.read(sheet_path))
    rows = [
        _row_values(row, shared)
        for row in root.findall(".//a:sheetData/a:row", XLSX_NS)
    ]
    header_index = _find_header_index(rows)
    if header_index is None:
        return []
    headers = [_normalize_header(value) for value in rows[header_index]]
    table = []
    for row in rows[header_index + 1 :]:
        record: dict[str, str] = {}
        for index, header in enumerate(headers):
            if not header:
                continue
            record[header] = row[index].strip() if index < len(row) else ""
        if any(record.values()):
            table.append(record)
    return table


def _row_values(row: ET.Element, shared: list[str]) -> list[str]:
    values: dict[int, str] = {}
    for cell in row.findall("a:c", XLSX_NS):
        index = _column_index(cell.attrib.get("r", ""))
        values[index] = _cell_value(cell, shared)
    if not values:
        return []
    return [values.get(index, "") for index in range(max(values) + 1)]


def _cell_value(cell: ET.Element, shared: list[str]) -> str:
    raw_node = cell.find("a:v", XLSX_NS)
    raw = raw_node.text if raw_node is not None else ""
    if cell.attrib.get("t") == "s" and raw.isdigit():
        index = int(raw)
        return shared[index] if index < len(shared) else raw
    inline = cell.find("a:is", XLSX_NS)
    if inline is not None:
        return "".join(node.text or "" for node in inline.findall(".//a:t", XLSX_NS))
    return raw or ""


def _column_index(reference: str) -> int:
    letters = re.match(r"([A-Z]+)", reference.upper())
    if not letters:
        return 0
    value = 0
    for char in letters.group(1):
        value = (value * 26) + (ord(char) - ord("A") + 1)
    return value - 1


def _find_header_index(rows: list[list[str]]) -> int | None:
    for index, row in enumerate(rows[:12]):
        normalized = {_normalize_header(value) for value in row}
        if (
            "ticker" in normalized
            or "route_id" in normalized
            or "candidate_id" in normalized
        ):
            return index
    return None


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _load_universe(
    tables: dict[str, list[dict[str, str]]],
) -> dict[str, CTCUniverseRow]:
    rows = tables.get("Full_Disposition_34") or tables.get("Company_Universe") or []
    universe: dict[str, CTCUniverseRow] = {}
    for raw in rows:
        ticker = _text(raw, "ticker").upper()
        if not ticker:
            continue
        universe[ticker] = CTCUniverseRow(
            ticker=ticker,
            company=_text(raw, "company"),
            sleeve=_text(raw, "sleeve", "category"),
            primary_mechanism=_text(raw, "primary_mechanism"),
            current_research_state=_text(
                raw, "current_research_state", "initial_research_bucket"
            ),
            preferred_vehicle=_text(raw, "preferred_vehicle"),
            preferred_structure=_text(raw, "preferred_structure"),
            disposition_score=_float(raw, "disposition_score_0_100", "thesis_score"),
            thesis_score=_float(raw, "thesis_score"),
            rank=_int(raw, "rank"),
        )
    if tables.get("Company_Universe"):
        _merge_company_universe(universe, tables["Company_Universe"])
    return universe


def _merge_company_universe(
    universe: dict[str, CTCUniverseRow], rows: list[dict[str, str]]
) -> None:
    for raw in rows:
        ticker = _text(raw, "ticker").upper()
        if not ticker or ticker not in universe:
            continue
        current = universe[ticker]
        universe[ticker] = CTCUniverseRow(
            ticker=ticker,
            company=current.company or _text(raw, "company"),
            sleeve=current.sleeve or _text(raw, "category"),
            primary_mechanism=current.primary_mechanism
            or _text(raw, "primary_mechanism"),
            current_research_state=current.current_research_state
            or _text(raw, "initial_research_bucket"),
            preferred_vehicle=current.preferred_vehicle,
            preferred_structure=current.preferred_structure,
            disposition_score=current.disposition_score,
            thesis_score=current.thesis_score or _float(raw, "thesis_score"),
            rank=current.rank,
        )


def _load_trade_rows(tables: dict[str, list[dict[str, str]]]) -> dict[str, CTCTradeRow]:
    records: dict[str, CTCTradeRow] = {}
    for sheet_name in ("I4_Playbook", "Trade_Triage", "Options_Candidates"):
        for raw in tables.get(sheet_name, []):
            ticker = _text(raw, "ticker").upper()
            if not ticker or ticker in records:
                continue
            records[ticker] = CTCTradeRow(
                ticker=ticker,
                structure=_text(
                    raw, "primary_structure", "structure", "structure_category"
                ),
                expiry=_text(raw, "expiry", "expiration"),
                dte=_int(raw, "dte"),
                net_debit_proxy=_float(raw, "proxy_debit_usd", "net_debit_proxy_usd"),
                max_loss_proxy=_float(
                    raw, "max_loss_per_spread_usd", "total_max_risk_usd"
                ),
            )
    return records


def _load_execution_checks(
    tables: dict[str, list[dict[str, str]]],
) -> dict[str, dict[str, str]]:
    checks: dict[str, dict[str, str]] = {}
    for raw in tables.get("I4_Execution_Check", []):
        ticker = _text(raw, "ticker").upper()
        if ticker:
            checks[ticker] = raw
    return checks


def _load_nerv_by_symbol(path: str | Path) -> dict[str, dict[str, Any]]:
    board_path = Path(path)
    if not board_path.exists():
        return {}
    payload = json.loads(board_path.read_text(encoding="utf-8"))
    by_symbol: dict[str, dict[str, Any]] = {}
    for contract in payload.get("contracts", []):
        symbol = str(contract.get("underlying") or "").upper()
        if symbol and symbol not in by_symbol:
            by_symbol[symbol] = contract
    return by_symbol


def _build_desk_row(
    ctc: CTCUniverseRow,
    trade: CTCTradeRow,
    execution_check: dict[str, str],
    nerv: dict[str, Any] | None,
) -> dict[str, Any]:
    priority = str((nerv or {}).get("manual_validation_priority") or "missing_nerv")
    flags = str((nerv or {}).get("rejection_flags") or "")
    fresh_quote = _text(execution_check, "fresh_quote").upper()
    strategy = classify_strategy(
        structure=trade.structure,
        preferred_structure=ctc.preferred_structure,
        preferred_vehicle=ctc.preferred_vehicle,
    )
    state, next_action = _desk_state(ctc, priority, flags, fresh_quote, strategy)
    block_reason = _block_reason(ctc, priority, flags, fresh_quote, strategy)
    return {
        "rank": ctc.rank,
        "ticker": ctc.ticker,
        "company": ctc.company,
        "sleeve": ctc.sleeve,
        "primary_mechanism": ctc.primary_mechanism,
        "current_research_state": ctc.current_research_state,
        "preferred_vehicle": ctc.preferred_vehicle,
        "preferred_structure": ctc.preferred_structure,
        "disposition_score": ctc.disposition_score,
        "thesis_score": ctc.thesis_score,
        "ctc_structure": trade.structure,
        "ctc_expiry": trade.expiry,
        "ctc_dte": trade.dte,
        "ctc_net_debit_proxy": trade.net_debit_proxy,
        "structure_family": strategy.family,
        "structure_complexity": strategy.complexity,
        "structure_taxonomy_reason": strategy.reason,
        "manual_complex_review_required": strategy.manual_complex_review_required,
        "nerv_best_contract": (nerv or {}).get("contract_symbol"),
        "nerv_best_expiration": (nerv or {}).get("expiration"),
        "nerv_best_option_type": (nerv or {}).get("option_type"),
        "nerv_best_strike": (nerv or {}).get("strike"),
        "nerv_score": (nerv or {}).get("nerv_score"),
        "nerv_priority": priority,
        "nerv_rejection_flags": flags,
        "desk_state": state,
        "execute_block_reason": block_reason,
        "next_action": next_action,
    }


def _desk_state(
    ctc: CTCUniverseRow,
    priority: str,
    flags: str,
    fresh_quote: str,
    strategy: StrategyClassification,
) -> tuple[str, str]:
    if "OPTION" not in ctc.preferred_vehicle.upper() and not ctc.preferred_structure:
        return (
            "equity_or_research_only",
            "Keep in regime/catalyst board; no option review yet.",
        )
    if priority == "missing_nerv":
        return (
            "needs_nerv_snapshot",
            "Run NERV for this ticker before structure review.",
        )
    if priority == "reject" or _has_fatal_flags(flags):
        return (
            "reject_options_for_now",
            "Reject/refresh option chain; inspect liquidity and contract existence.",
        )
    if priority == "refresh" or fresh_quote != "YES":
        return (
            "refresh_quote_required",
            "Refresh at broker/Cboe; final debit and size remain blocked.",
        )
    if strategy.manual_complex_review_required:
        return (
            "manual_complex_structure_review",
            "Model complex payoff, margin, assignment/dividend risk, and broker quote.",
        )
    return (
        "manual_validate_candidate",
        "Manually validate spread geometry, catalyst gate, and broker quote.",
    )


def _block_reason(
    ctc: CTCUniverseRow,
    priority: str,
    flags: str,
    fresh_quote: str,
    strategy: StrategyClassification,
) -> str:
    reasons = ["fresh_broker_quote_required"]
    if ctc.ticker != "SPY":
        reasons.append("non_spy_requires_expanded_operator_risk_policy")
    if priority in {"missing_nerv", "reject", "refresh"}:
        reasons.append(f"nerv_priority_{priority}")
    if flags:
        reasons.append(f"nerv_flags_{flags}")
    if strategy.manual_complex_review_required:
        reasons.append("complex_structure_manual_review_required")
    if strategy.complexity == "branch_pending":
        reasons.append("structure_branch_not_finalized")
    if fresh_quote != "YES":
        reasons.append("ctc_execution_check_fresh_quote_not_yes")
    return ";".join(reasons)


def _has_fatal_flags(flags: str) -> bool:
    fatal = {"missing_bid_ask", "crossed_market", "missing_midpoint", "no_activity"}
    return any(flag in fatal for flag in flags.split(";"))


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_state: dict[str, int] = {}
    symbols_by_state: dict[str, list[str]] = {}
    for row in rows:
        state = row["desk_state"]
        ticker = row["ticker"]
        by_state[state] = by_state.get(state, 0) + 1
        symbols_by_state.setdefault(state, []).append(ticker)
    return {
        "row_count": len(rows),
        "states": by_state,
        "symbols_by_state": symbols_by_state,
        "manual_validate_count": by_state.get("manual_validate_candidate", 0),
        "all_execution_blocked_until_broker_quote": True,
        "suggested_nerv_symbols": sorted(
            set(symbols_by_state.get("needs_nerv_snapshot", []))
        ),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CTC/NERV Trade Desk Board",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "**Mode:** research-only, human-in-the-loop. No order authority is granted.",
        "",
        "## Summary",
        "",
        f"- Rows: {payload['summary']['row_count']}",
        f"- Manual-validate candidates: {payload['summary']['manual_validate_count']}",
        "- Execution: blocked until fresh broker quote and explicit operator approval.",
        "- Suggested NERV symbols: "
        + (", ".join(payload["summary"].get("suggested_nerv_symbols", [])) or "none"),
        "",
        "## Top Rows",
        "",
        "| Rank | Ticker | Company | State | Structure | NERV | Next action |",
        "|---:|---|---|---|---|---:|---|",
    ]
    for row in payload["rows"][:20]:
        lines.append(
            "| {rank} | {ticker} | {company} | {state} | {structure} | {score} | {next_action} |".format(
                rank=row.get("rank") or "",
                ticker=row["ticker"],
                company=_md(row.get("company")),
                state=row["desk_state"],
                structure=_md(
                    row.get("ctc_structure") or row.get("preferred_structure")
                ),
                score=row.get("nerv_score") or "",
                next_action=_md(row["next_action"]),
            )
        )
    lines.extend(["", "## Governance", "", payload["governance"]["warning"], ""])
    return "\n".join(lines)


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|")


def _sort_key(row: CTCUniverseRow) -> tuple[int, float, str]:
    rank = row.rank if row.rank is not None else 9999
    score = -(row.disposition_score or row.thesis_score or 0)
    return (rank, score, row.ticker)


def _text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _float(row: dict[str, Any], *keys: str) -> float | None:
    text = _text(row, *keys).replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _int(row: dict[str, Any], *keys: str) -> int | None:
    value = _float(row, *keys)
    return int(value) if value is not None else None
