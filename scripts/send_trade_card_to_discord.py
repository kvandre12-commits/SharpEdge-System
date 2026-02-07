import json
import urllib.request
import urllib.error
import textwrap

MAX = 1900  # keep buffer under 2000

def _post(content: str):
    payload = json.dumps({"content": content}).encode("utf-8")
    req = urllib.request.Request(
        WEBHOOK,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "SharpEdge-System/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Discord HTTP {e.code}: {body}") from e

def send(msg: str):
    msg = msg.strip()
    if len(msg) <= MAX:
        return _post(msg)

    # Split on lines first (cleaner), then hard-wrap as fallback
    chunks = []
    buf = ""
    for line in msg.splitlines(True):  # keep line breaks
        if len(buf) + len(line) <= MAX:
            buf += line
        else:
            if buf:
                chunks.append(buf.rstrip())
            # if a single line is huge, wrap it
            if len(line) > MAX:
                chunks.extend(textwrap.wrap(line, width=MAX))
                buf = ""
            else:
                buf = line
    if buf:
        chunks.append(buf.rstrip())

    for i, c in enumerate(chunks, 1):
        prefix = f"**(part {i}/{len(chunks)})**\n" if len(chunks) > 1 else ""
        _post(prefix + c)
