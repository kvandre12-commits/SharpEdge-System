"""Truth Social event scanner for SharpEdge cockpit.

This is event context, not execution authority.  The scanner is deliberately
fail-soft because Truth Social's public API is often Cloudflare-protected from
headless clients.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import re
from pathlib import Path
from typing import Any

TRUMP_ACCOUNT_ID = "107780257626128497"
TRUTH_SOCIAL_STATUSES_URL = (
    "https://truthsocial.com/api/v1/accounts/"
    f"{TRUMP_ACCOUNT_ID}/statuses?limit=20&exclude_replies=true&only_media=false"
)
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_PATH = ROOT_DIR / "outputs" / "trump_truth_event_scan.json"
DEFAULT_TEXT_PATH = ROOT_DIR / "outputs" / "trump_truth_event_scan.txt"
DEFAULT_MANUAL_PATH = ROOT_DIR / "data" / "trump_truth_manual.json"
CACHE_TTL_SECONDS = 300
MAX_TEXT_CHARS = 420

EVENT_KEYWORDS = {
    "fed_rates": (
        "fed",
        "fomc",
        "powell",
        "rate",
        "rates",
        "interest",
        "inflation",
        "cpi",
        "ppi",
        "treasury",
        "bond",
        "dollar",
    ),
    "trade_tariffs": (
        "tariff",
        "tariffs",
        "china",
        "trade deal",
        "trade war",
        "import",
        "exports",
    ),
    "geopolitics": (
        "iran",
        "israel",
        "russia",
        "ukraine",
        "nato",
        "war",
        "sanctions",
        "oil",
        "crude",
    ),
    "growth_jobs": (
        "jobs",
        "employment",
        "unemployment",
        "gdp",
        "recession",
        "manufacturing",
        "consumer",
    ),
    "policy_tax": (
        "tax",
        "taxes",
        "spending",
        "debt ceiling",
        "shutdown",
        "budget",
        "chips",
        "ai",
    ),
}

HIGH_IMPACT_TERMS = {
    "fomc",
    "powell",
    "tariff",
    "tariffs",
    "china",
    "iran",
    "oil",
    "crude",
    "inflation",
    "jobs",
    "shutdown",
    "sanctions",
}

TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _parse_time(value: Any) -> dt.datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = dt.datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except ValueError:
        return None


def _strip_html(value: Any) -> str:
    text = TAG_RE.sub(" ", str(value or ""))
    return SPACE_RE.sub(" ", html.unescape(text)).strip()


def _truncate(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    clean = SPACE_RE.sub(" ", text).strip()
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 1)].rstrip() + "…"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _cache_is_fresh(
    payload: dict[str, Any], now: dt.datetime, ttl_seconds: int
) -> bool:
    fetched = _parse_time(payload.get("fetched_at_utc"))
    if fetched is None:
        return False
    return (now - fetched).total_seconds() <= ttl_seconds


def _normalize_status(raw: dict[str, Any]) -> dict[str, Any]:
    content = _strip_html(raw.get("content") or raw.get("text") or raw.get("body"))
    created_at = str(raw.get("created_at") or raw.get("createdAt") or "")
    url = str(raw.get("url") or raw.get("uri") or raw.get("link") or "")
    status_id = str(raw.get("id") or raw.get("status_id") or raw.get("statusId") or "")
    return {
        "id": status_id,
        "created_at": created_at,
        "url": url,
        "text": _truncate(content),
    }


def _normalize_status_list(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict) and isinstance(raw.get("statuses"), list):
        raw = raw["statuses"]
    if isinstance(raw, dict) and isinstance(raw.get("items"), list):
        raw = raw["items"]
    if not isinstance(raw, list):
        return []
    statuses = [_normalize_status(item) for item in raw if isinstance(item, dict)]
    return [item for item in statuses if item.get("text")]


def _term_matches(term: str, text: str) -> bool:
    pattern = re.escape(term.lower()).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", text) is not None


def _classify_status(status: dict[str, Any], now: dt.datetime) -> dict[str, Any]:
    text = str(status.get("text") or "")
    lower = text.lower()
    categories: list[str] = []
    matched_terms: list[str] = []
    for category, terms in EVENT_KEYWORDS.items():
        hits = [term for term in terms if _term_matches(term, lower)]
        if hits:
            categories.append(category)
            matched_terms.extend(hits)
    created = _parse_time(status.get("created_at"))
    age_hours = None
    if created is not None:
        age_hours = round(max(0.0, (now - created).total_seconds() / 3600.0), 2)
    high_hits = sorted({term for term in matched_terms if term in HIGH_IMPACT_TERMS})
    base_score = (
        len(set(matched_terms)) * 12 + len(categories) * 8 + len(high_hits) * 10
    )
    if age_hours is not None:
        if age_hours <= 2:
            base_score += 25
        elif age_hours <= 12:
            base_score += 12
        elif age_hours > 72:
            base_score -= 15
    score = max(0, min(100, base_score))
    if score >= 55 or high_hits:
        impact = "high"
    elif score >= 25:
        impact = "medium"
    elif score > 0:
        impact = "low"
    else:
        impact = "none"
    enriched = dict(status)
    enriched.update(
        {
            "market_relevant": bool(matched_terms),
            "event_score": score,
            "impact": impact,
            "categories": categories,
            "matched_terms": sorted(set(matched_terms)),
            "age_hours": age_hours,
        }
    )
    return enriched


def _fetch_truth_social_statuses(
    timeout_seconds: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import requests
    except (
        ImportError
    ) as exc:  # pragma: no cover - requests is installed in cockpit env
        return [], {"ok": False, "status": "requests_missing", "error": str(exc)}
    try:
        response = requests.get(
            TRUTH_SOCIAL_STATUSES_URL,
            headers={"User-Agent": "Mozilla/5.0 SharpEdge Truth Scanner"},
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        return [], {"ok": False, "status": "request_failed", "error": str(exc)}
    if response.status_code != 200:
        return [], {
            "ok": False,
            "status": "source_blocked"
            if response.status_code in {403, 429}
            else "bad_status",
            "http_status": response.status_code,
            "content_type": response.headers.get("content-type"),
        }
    try:
        return _normalize_status_list(response.json()), {
            "ok": True,
            "status": "ok",
            "http_status": response.status_code,
        }
    except (json.JSONDecodeError, ValueError) as exc:
        return [], {"ok": False, "status": "bad_json", "error": str(exc)}


def _summary(latest: dict[str, Any] | None, source_status: str) -> tuple[str, str]:
    if latest:
        impact = str(latest.get("impact") or "event")
        cats = ", ".join(latest.get("categories") or ["event"])
        return (
            f"Trump Truth event watch: {impact.upper()} {cats}",
            str(latest.get("text") or ""),
        )
    if source_status == "source_blocked":
        return (
            "Trump Truth scanner blocked by source",
            "Truth Social API blocked the headless scanner; use manual override if the browser shows a market-moving Truth.",
        )
    return (
        "No market-relevant Trump Truth found",
        "Scanner did not find a recent Truth containing tracked macro/event terms.",
    )


def build_truth_social_event_scan(
    *,
    now: dt.datetime | None = None,
    cache_path: Path | str | None = None,
    manual_path: Path | str | None = None,
    text_path: Path | str | None = None,
    ttl_seconds: int = CACHE_TTL_SECONDS,
    enable_network: bool = True,
) -> dict[str, Any]:
    """Return latest market/event-relevant Trump Truth scan.

    Manual override format: either a list of status dicts or {"statuses": [...]}.
    Each status may contain content/text/body plus created_at and url.
    """
    now = now or _now_utc()
    cache = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
    manual = Path(manual_path) if manual_path else DEFAULT_MANUAL_PATH
    text_output = Path(text_path) if text_path else DEFAULT_TEXT_PATH

    if manual.exists():
        raw_statuses = _normalize_status_list(_load_json(manual))
        source = {"ok": True, "status": "manual_override", "path": str(manual)}
    elif cache.exists():
        try:
            cached = _load_json(cache)
            if isinstance(cached, dict) and _cache_is_fresh(cached, now, ttl_seconds):
                return cached
        except (json.JSONDecodeError, OSError):
            pass
        raw_statuses, source = ([], {"ok": False, "status": "network_disabled"})
        if enable_network:
            raw_statuses, source = _fetch_truth_social_statuses()
    elif enable_network:
        raw_statuses, source = _fetch_truth_social_statuses()
    else:
        raw_statuses, source = ([], {"ok": False, "status": "network_disabled"})

    enriched = [_classify_status(status, now) for status in raw_statuses]
    relevant = [item for item in enriched if item.get("market_relevant")]
    relevant.sort(
        key=lambda item: (
            int(item.get("event_score") or 0),
            item.get("created_at") or "",
        ),
        reverse=True,
    )
    latest_relevant = relevant[0] if relevant else None
    latest_any = enriched[0] if enriched else None
    headline, story = _summary(latest_relevant, str(source.get("status") or "unknown"))
    payload = {
        "schema": "sharpedge.truth_social_event_scan.v1",
        "available": bool(source.get("ok")) and bool(enriched),
        "fetched_at_utc": now.isoformat(timespec="seconds"),
        "account": "@realDonaldTrump",
        "headline": headline,
        "story": story,
        "source": "truthsocial:realDonaldTrump",
        "source_status": source,
        "latest_relevant": latest_relevant,
        "latest_any": latest_any,
        "relevant_count": len(relevant),
        "status_count": len(enriched),
        "items": relevant[:5],
        "research_only_warning": "Social catalyst context only; does not override graph, spine, or operator approval.",
    }
    try:
        _write_json(cache, payload)
        write_truth_social_event_scan_text(payload, text_output)
    except OSError:
        pass
    return payload


def write_truth_social_event_scan_text(
    payload: dict[str, Any], path: Path | str = DEFAULT_TEXT_PATH
) -> None:
    latest = payload.get("latest_relevant") or {}
    lines = [
        "# Trump Truth Event Scan",
        "",
        str(payload.get("headline") or ""),
        "",
        str(payload.get("story") or ""),
        "",
        f"source_status: {(payload.get('source_status') or {}).get('status')}",
    ]
    if latest:
        lines.extend(
            [
                f"created_at: {latest.get('created_at')}",
                f"impact: {latest.get('impact')} score={latest.get('event_score')}",
                f"terms: {', '.join(latest.get('matched_terms') or [])}",
                f"url: {latest.get('url')}",
            ]
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_CACHE_PATH",
    "DEFAULT_MANUAL_PATH",
    "TRUTH_SOCIAL_STATUSES_URL",
    "build_truth_social_event_scan",
    "write_truth_social_event_scan_text",
]
