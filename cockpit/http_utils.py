from __future__ import annotations

import time
from typing import Any

import requests

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def request_json_with_backoff(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 20,
    attempts: int = 3,
    base_sleep_seconds: float = 1.0,
    retryable_status_codes: set[int] | None = None,
) -> Any:
    """GET JSON with tiny retry/backoff discipline for flaky public endpoints."""
    retryable = retryable_status_codes or RETRYABLE_STATUS_CODES
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            if response.status_code in retryable and attempt < attempts:
                time.sleep(base_sleep_seconds * (2 ** (attempt - 1)))
                continue
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            last_error = exc
            if status_code not in retryable or attempt >= attempts:
                raise
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt >= attempts:
                raise
        time.sleep(base_sleep_seconds * (2 ** (attempt - 1)))

    if last_error is not None:
        raise last_error
    raise RuntimeError(
        f"request_json_with_backoff exhausted without response for {url}"
    )
