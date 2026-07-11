# SharpEdge Agile Dashboard Stack

This document defines the actual card stack for an agile SharpEdge dashboard.

It lives inside the viewing layer. That means:

```text
The dashboard may adapt what it foregrounds.
The dashboard may not adapt what the market means.
```

See also:

- `docs/architecture/VIEWING_LAYER.md`
- `docs/operator_review_and_dashboard.md`

## Design goals

1. Put the operator's **next useful question** first.
2. Make action urgency visible without inventing authority.
3. Preserve one-way flow from interpretation -> authority -> viewing.
4. Reuse completed protocols instead of spawning dashboard-only logic.

## The 40/70 decision rule

Dashboard design should foreground three decision bands:

- **0-39** -> **Observe**
  - not enough certainty to lean
  - show blockers, freshness, and context
- **40-69** -> **Can Act**
  - discretionary action zone
  - show setup, authority, invalidation, and execution burden clearly
- **70-100** -> **Should Act**
  - action-required zone
  - do not bury the trigger, authority, or invalidation details

Important:

```text
The dashboard may emphasize these bands.
The dashboard may not manufacture them from vibes.
```

Use already-completed upstream authority/permission outputs.

## Card inventory from current surfaces

### Current operator-surface cards

- current state
- live cockpit snapshot
- execution state
- artifact freshness
- recent work
- connector surface
- watchlist
- latest operator journal
- workflow state
- approval state
- operator brief
- blocking reasons
- risk flags
- next steps

### Current live-read / cockpit cards

- setup conviction
- market day / bucket brain
- bucket-conditioned spine
- permission score trend
- edge token engine
- market behavior annotations
- setup context / weekly / monthly context

## Global composition rules

### Sticky cards

Cards that stay visible or near-visible in most modes:

1. **Decision band / authority banner**
2. **Primary market state card**
3. **Blocking reasons / risk posture**
4. **Freshness / proof status**

### Collapsible cards

Cards that can drop lower or collapse behind toggles:

- recent work
- full workflow state
- raw approval state dump
- journal history older than latest review slice
- secondary context charts when a live event is active

### Escalation rule

When the surface enters a stronger action band, the dashboard should:

- move authority upward
- move invalidation upward
- reduce peripheral cards
- shorten narrative text
- make active setup / event / commitment impossible to miss

## Mode 1 - Open

Question:

```text
What battlefield are we opening into, and are we operationally ready?
```

### P0 (always top)

1. **Decision band / readiness banner**
   - source: `workflow_state`, `approval_decision`, `operator_brief`
   - shows: observe / can act / should act, readiness, operator action
2. **Battlefield card**
   - source: `signal.trade_permission.market_day` or equivalent bucket packet
   - shows: regime, bucket, risk posture, allowed playbooks
3. **Artifact freshness card**
   - source: output mtimes / render proof
   - shows: stale vs fresh inputs
4. **Opening checklist**
   - source: `morning_open_dashboard.json`
   - shows: blockers, permissions, operational checks

### P1 (visible by default)

5. **Key levels + open focus**
   - source: signal / brief / watchlist
   - shows: spot, gap direction, gap fill, OR/PD references, ATM, dealer hint
6. **Approval blockers / risk flags**
   - source: `approval_decision`, `operator_brief`
7. **Top watchlist focus**
   - source: `operator_watchlist.json`

### P2 (collapsed or lower)

8. **Workflow state**
9. **Operator brief**
10. **Journal context / historical hints**
11. **Recent work**

### Open-mode collapse rule

If readiness is blocked or inputs are stale:

- expand freshness
- expand blockers
- collapse recent work
- collapse journal/history below the fold

## Mode 2 - Live

Question:

```text
What just happened, did the market commit, and do we act now?
```

### P0 (always top)

1. **Decision band / authority banner**
   - source: trade permission / authority outputs
   - shows: observe / can act / should act
2. **Liquidity event + setup headline**
   - source: setup/event outputs
   - shows: failed break, breakout, exhaustion, trap, reclaim, etc.
3. **Acceptance / commitment card**
   - source: acceptance-facing packet / score reason
   - shows: accepted / failed / unresolved commitment state
4. **Pressure state card**
   - source: pressure state / pressure score / trend context
   - shows: coiled / unresolved / release / normal
5. **Invalidation / risk card**
   - source: approval / setup / execution burden outputs
   - shows: what breaks the thesis

### P1 (visible by default)

6. **Bucket brain**
   - source: market day packet
7. **Bucket-conditioned spine**
   - source: spine packet
8. **Setup conviction**
   - source: `trade_permission.setup_conviction`
9. **Execution state / edge token engine**
   - source: beta execution / edge token packets
10. **Permission score trend**
   - source: permission trend packet

### P2 (collapsed or lower)

11. **Behavior annotations**
12. **Weekly / monthly context**
13. **Connector surface**
14. **Journal slice**

### Live-mode collapse rule

When a live liquidity event is active and authority is >= 70:

- keep event, acceptance, pressure, authority, invalidation on screen
- collapse review/history cards
- collapse raw workflow/admin cards
- do not let bucket/spine push acceptance off screen

## Mode 3 - Review

Question:

```text
What changed, what mattered, and what blocked or confirmed action?
```

### P0 (always top)

1. **Review headline / outcome banner**
   - source: session review / journal / decision receipts
2. **Decision band history**
   - source: permission trend / authority snapshots
3. **What changed card**
   - source: permission trend / setup lifecycle / behavior annotations
4. **Blocked vs acted summary**
   - source: journal / approval / review outputs

### P1 (visible by default)

5. **Setup lifecycle card**
6. **Top blockers and risk flags**
7. **Journal highlights**
8. **Market behavior annotations**
9. **Receipt / proof references**

### P2 (collapsed or lower)

10. **Raw workflow state**
11. **Connector implementation detail**
12. **Recent work / git state**

### Review-mode collapse rule

When no action occurred:

- expand blockers and invalidation analysis
- shrink execution-state detail
- foreground missed/withheld action reasoning

## Mode 4 - Handoff

Question:

```text
Can another surface or system render the same truth without drift?
```

### P0 (always top)

1. **Compact authority snapshot**
   - source: canonical signal / trade permission contract
2. **Transport integrity card**
   - source: request/view-model/export traces
3. **Render proof card**
   - source: prelaunch trace / viewer observation
4. **Freshness + source-of-truth card**
   - shows canonical artifact path and timestamps

### P1 (visible by default)

5. **Bucket / setup / bias compact summary**
6. **Expected target URL / target viewer**
7. **Observation result / render status**

### P2 (collapsed or lower)

8. **Deep explanation blocks**
9. **Journal / workflow context**
10. **Recent work**

### Handoff-mode collapse rule

If the handoff path is degraded:

- expand transport integrity
- expand render proof
- collapse market storytelling below the fold

## Source map

| Card family | Upstream source |
|---|---|
| Decision band / authority | `trade_permission`, `approval_decision`, `workflow_state` |
| Battlefield / bucket | `market_day`, `day_bucket`, regime outputs |
| Liquidity event / setup | setup detector outputs, `entry_setup_tag`, lifecycle packets |
| Acceptance | acceptance score/reason or dedicated commitment packet |
| Pressure state | pressure state / trend / pressure reason |
| Invalidation / blockers | `approval_decision`, setup invalidation, execution burden |
| Spine / score trend | bucket-conditioned spine, permission trend |
| Execution state | edge token / beta execution packets |
| Freshness / proof | mtimes, prelaunch trace, viewer observation |
| Review / journal | session review, operator journal, decision receipts |

## Non-goals

This dashboard must not:

- create a new trade signal
- override authority
- hide blockers because confidence is high
- infer acceptance from a prettier trend line
- turn score summaries into independent truth
- let handoff/mobile views drift from canonical source contracts

## Webhook-shaped mental model

If you usually spin this up from a webhook, think of it like this:

### Simple GET shape

```text
GET /dashboard?mode=open
GET /dashboard?mode=live
GET /dashboard?mode=review
GET /dashboard?mode=handoff
```

### Simple POST shape

```json
{
  "mode": "live",
  "surface": "operator",
  "symbol": "SPY"
}
```

### What the mode means

- `open` -> foreground checklist, freshness, blockers, battlefield
- `live` -> foreground event, acceptance, pressure, authority
- `review` -> foreground changes, blockers, receipts, journal context
- `handoff` -> foreground compact summary, transport integrity, render proof

### What does not change

The webhook does **not** send new market meaning.
It only asks the viewing layer which completed cards to put on top.

### Response mental model

The response can still be the same kind of thing you already use:

```json
{
  "mode": "live",
  "surface": "operator",
  "artifact": "cockpit/operator_surface.html",
  "url": "http://127.0.0.1:8777/operator_surface.html"
}
```

Or, if you are driving a mobile/view-model lane:

```json
{
  "mode": "handoff",
  "surface": "phone_companion",
  "artifact": "phone_companion/views/trading/golden_loop_view_model.json"
}
```

### Auto mode fallback

If the webhook does not specify `mode`, the renderer may choose a default:

- premarket / market-open prep -> `open`
- live session + fresh signal -> `live`
- post-session or explicit review request -> `review`
- export/share/import path -> `handoff`

That is all "mode selector" means here: webhook chooses viewing posture.
Not market logic. Not authority logic.

## Recommendation for implementation order

1. Add an explicit **dashboard mode selector** to the operator surface.
2. Add a **decision band banner** implementing the 40/70 emphasis rule.
3. Recompose existing cards into the four mode stacks before inventing any new card.
4. Only add new upstream packets if a mode has a genuine hole.

That keeps this DRY, monotonic, and not stupid.
