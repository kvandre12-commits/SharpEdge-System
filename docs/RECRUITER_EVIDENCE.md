# SharpEdge: 60-Second Engineering Evidence

## What this is

SharpEdge is Kurtis Van Dress's Python and SQLite market-data and
human-governed agent system. It is designed to keep three responsibilities
separate:

```text
Observation -> Learning -> Authorization
```

A learning result can become review-eligible. It cannot grant itself policy,
approval, broker, or execution authority.

## Who built what

- **Kurtis Van Dress** owns the problem, system direction, operational judgment,
  and consequential authority.
- **SharpEdge** is the system being built and tested in this repository.
- **Code Puppy** is part of the AI-assisted development workflow used to inspect,
  implement, test, and document bounded changes. It is not the trading system and
  does not hold human accountability.

## Strongest externally verified contribution

Kurtis authored
[Code Puppy PR #885](https://github.com/mpfaffenberger/code_puppy/pull/885),
`fix(messaging): complete request futures on owning event loop`.

The public record shows:

- status: **merged** on 2026-09-22;
- author: `kvandre12-commits`;
- scope: `code_puppy/messaging/bus.py` and `tests/messaging/test_bus.py`;
- change size: 109 additions and 44 deletions;
- public quality, macOS Python 3.13, and Windows-encoding checks passed;
- upstream merge commit: `22f93115ca529e29d9c67b1249bbde79490eda3b`.

The fix preserved each request future together with its owning event loop and
completed that future on the correct loop. The linked PR contains the code,
tests, review history, and CI evidence.

## SharpEdge recovery proof

The recovery demonstration is deliberately synthetic and brokerless. It does
not claim live trading or autonomous order authority.

### Process interruption

The test starts a real child process that durably records three settled learning
candidates, opens review publication, verifies and recomputes the conclusion,
and then terminates with `os._exit(86)` before commit.

State immediately after death:

```text
candidates = 3
reviews    = 0
projection = absent
```

A fresh process then:

1. reuses all three immutable candidates;
2. recomputes the review from SQLite evidence;
3. publishes exactly one review;
4. writes the rebuildable JSON projection.

A third replay inserts nothing:

```text
candidates = 3
reviews    = 1
new candidate records = 0
new review records    = 0
```

### Trust checks

The ledger rejects:

- the same cycle identity with changed candidate content;
- duplicate cycle evidence used to inflate confidence;
- a review whose source receipts differ from staged evidence;
- a conclusion that cannot be reproduced byte-for-byte from stored candidates.

Even a correctly referenced review with a falsified aggregate value is rejected
without writing a review record.

## Reproduce it

From this branch:

```bash
python3 -m pytest \
  tests/test_recoverable_compounding_demo.py \
  tests/test_recoverable_cycle_demo.py \
  tests/test_deterministic_cycle_demo.py \
  tests/test_disagreement_adjudicator.py -q
```

Expected focused result:

```text
22 passed
```

Run the crash boundary directly:

```bash
STATE="$(mktemp -d)"
python3 scripts/agents/recoverable_compounding_demo.py \
  --state-dir "$STATE" \
  --output "$STATE/review.json" \
  --crash-before-review
```

Expected exit code: `86`. Restart without `--crash-before-review`, then run the
same restart command again. The first restart publishes one review; the replay
publishes none.

## Relevant commits

```text
cb5c1a0  Add deterministic protective cycle demo
b5907fb  Prove cycle recovery across publication crash
d26304e  Aggregate compounding evidence without self-promotion
4d94229  Recover compounding review publication exactly once
```

These commit links become publicly verifiable only after this feature branch is
published to GitHub. Until then, describe the recovery result as locally tested,
not publicly inspectable.

## Honest boundary

This evidence demonstrates deterministic synthetic execution, immutable
learning evidence, actual process-interruption recovery, reproducible review
construction, and idempotent replay.

It does **not** demonstrate live-market recovery, production broker recovery,
autonomous trading, profitable performance, or permission for an AI system to
approve its own recommendations.
