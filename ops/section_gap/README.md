# Section Gap Ops Pilot

This folder is the home for the **store-floor empty section / fresh monitoring pilot**.

It exists specifically to keep this work **out of** the trading-only `cockpit/` lane.

## Purpose

Use real reference photos, observed photos, and a simple routing model to:
- classify one pilot section
- identify likely owner
- package an alert
- render a focused operator dashboard
- build a repeat-event history

## Layer responsibilities

- `pilot_config.example.json` — pilot spot definition and routing metadata
- `refs/` — known-good reference images
- `observations/` — observed images for the pilot
- `events/` — generated section-gap events and history
- future `build_section_signal.py` — decision artifact builder
- future `build_section_dashboard.py` — dashboard renderer
- future `run_section_dashboard.sh` — local server + Brave launcher

## Non-goals

This folder does **not** contain:
- market data logic
- SPY / options / gamma logic
- Robinhood or trade decision logic

If that stuff shows up here, somebody fed the repo after midnight.

## Pilot philosophy

Start with one trustworthy spot.
A boring true alert beats a magical liar.
