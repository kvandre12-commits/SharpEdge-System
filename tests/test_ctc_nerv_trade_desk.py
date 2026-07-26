from __future__ import annotations

import json
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nerv.ctc_trade_desk import (  # noqa: E402
    build_trade_desk_payload,
    load_xlsx_tables,
    write_trade_desk_artifacts,
)


def test_load_xlsx_tables_reads_disposition_sheet(tmp_path: Path) -> None:
    workbook = tmp_path / "ctc.xlsx"
    _write_minimal_ctc_workbook(workbook)

    tables = load_xlsx_tables(workbook)

    assert tables["Full_Disposition_34"][0]["ticker"] == "BKR"
    assert (
        tables["Full_Disposition_34"][0]["preferred_structure"]
        == "Aug 55/60 call debit spread"
    )
    assert tables["I4_Playbook"][0]["proxy_debit_usd"] == "1.994"


def test_build_trade_desk_payload_joins_ctc_and_nerv(tmp_path: Path) -> None:
    workbook = tmp_path / "ctc.xlsx"
    nerv_board = tmp_path / "nerv_board.json"
    _write_minimal_ctc_workbook(workbook)
    nerv_board.write_text(
        json.dumps(
            {
                "schema": "sharpedge.nerv_liquidity_board.v1",
                "contracts": [
                    {
                        "underlying": "BKR",
                        "contract_symbol": "BKR260821C00055000",
                        "expiration": "2026-08-21",
                        "option_type": "call",
                        "strike": 55,
                        "nerv_score": 82.5,
                        "manual_validation_priority": "high",
                        "rejection_flags": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = build_trade_desk_payload(
        ctc_workbook=workbook,
        nerv_board_path=nerv_board,
        generated_at="2026-07-26T00:00:00+00:00",
    )

    row = payload["rows"][0]
    assert payload["schema"] == "sharpedge.ctc_nerv_trade_desk.v1"
    assert row["ticker"] == "BKR"
    assert row["ctc_structure"] == "Aug 55/60 Call Debit Spread"
    assert row["nerv_best_contract"] == "BKR260821C00055000"
    assert row["desk_state"] == "refresh_quote_required"
    assert "fresh_broker_quote_required" in row["execute_block_reason"]
    assert (
        "non_spy_requires_expanded_operator_risk_policy" in row["execute_block_reason"]
    )


def test_write_trade_desk_artifacts(tmp_path: Path) -> None:
    payload = {
        "schema": "sharpedge.ctc_nerv_trade_desk.v1",
        "generated_at": "2026-07-26T00:00:00+00:00",
        "summary": {"row_count": 1, "manual_validate_count": 0},
        "governance": {"warning": "research-only warning"},
        "rows": [
            {
                "rank": 14,
                "ticker": "BKR",
                "company": "Baker Hughes",
                "desk_state": "refresh_quote_required",
                "ctc_structure": "Aug 55/60 Call Debit Spread",
                "nerv_score": 82.5,
                "next_action": "Refresh at broker/Cboe.",
            }
        ],
    }

    paths = write_trade_desk_artifacts(payload, tmp_path)

    assert Path(paths["json"]).exists()
    assert "BKR" in Path(paths["csv"]).read_text(encoding="utf-8")
    assert "CTC/NERV Trade Desk Board" in Path(paths["markdown"]).read_text(
        encoding="utf-8"
    )


def _write_minimal_ctc_workbook(path: Path) -> None:
    sheets = {
        "Full_Disposition_34": [
            ["CTC-C001 | Full 34-Name Disposition Ledger v0.5"],
            [],
            [],
            [
                "Rank",
                "Ticker",
                "Company",
                "Sleeve",
                "Primary_Mechanism",
                "Current_Research_State",
                "Preferred_Vehicle",
                "Disposition_Score_0_100",
                "Preferred_Structure",
            ],
            [
                "14",
                "BKR",
                "Baker Hughes",
                "Oilfield / LNG equipment",
                "Energy infrastructure transmission",
                "Option structure populated",
                "OPTIONS",
                "71",
                "Aug 55/60 call debit spread",
            ],
        ],
        "Company_Universe": [
            ["Company_ID", "Company", "Ticker", "Category", "Thesis_Score"],
            ["CTC-COMP014", "Baker Hughes", "BKR", "Oilfield services", "4.2"],
        ],
        "I4_Playbook": [
            [
                "Priority",
                "Ticker",
                "Primary_Structure",
                "Expiry",
                "DTE",
                "Proxy_Debit_USD",
            ],
            ["1", "BKR", "Aug 55/60 Call Debit Spread", "2026-08-21", "28", "1.994"],
        ],
        "I4_Execution_Check": [
            ["Play_ID", "Ticker", "Fresh_Quote"],
            ["I4-BKR-01", "BKR", "NO"],
        ],
    }
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types(len(sheets)))
        archive.writestr("_rels/.rels", _root_rels())
        archive.writestr("xl/workbook.xml", _workbook_xml(list(sheets)))
        archive.writestr("xl/_rels/workbook.xml.rels", _workbook_rels(len(sheets)))
        for index, rows in enumerate(sheets.values(), start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _sheet_xml(rows))


def _content_types(sheet_count: int) -> str:
    overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        f"{overrides}</Types>"
    )


def _root_rels() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )


def _workbook_xml(sheet_names: list[str]) -> str:
    sheet_xml = "".join(
        f'<sheet name="{name}" sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheet_xml}</sheets></workbook>"
    )


def _workbook_rels(sheet_count: int) -> str:
    rels = "".join(
        f'<Relationship Id="rId{i}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{rels}</Relationships>"
    )


def _sheet_xml(rows: list[list[str]]) -> str:
    rows_xml = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for col_index, value in enumerate(row, start=1):
            ref = f"{_column_name(col_index)}{row_index}"
            cells.append(
                f'<c r="{ref}" t="inlineStr"><is><t>{_escape(value)}</t></is></c>'
            )
        rows_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(rows_xml)}</sheetData></worksheet>"
    )


def _column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
