# SharpEdge Live Cockpit

Real-time, data-driven market reads rendered in the phone browser (Brave via
Termux). Goal: **reduce time-to-execution** -- you SEE the move, the cockpit
CONFIRMS it with numbers in the same glance.

No paid APIs, no auth, no broker credentials. Free sources only:

- **Yahoo 1-min intraday** -> price action, VWAP control, momentum, volume
- **CBOE delayed options** -> open-interest walls, put/call ratio, ATM IV
  (includes greeks: delta/gamma/vega/theta)

## Scripts

| Script | What it makes |
|--------|---------------|
| `make_cockpit.py` | The live read: `cockpit.html` + `cockpit_chart.svg`. The main event. |
| `make_price_volume.py` | Continuous 2-week price + volume chart (`spy_price_volume.svg`) |
| `make_options.py` | Open-interest walls + IV skew (`spy_options.svg`) |
| `make_overlay.py` | Normalized intraday overlay of recent days (`spy_overlay.svg`) |

All scripts are stdlib + `requests` only, and hand-render SVG (no matplotlib /
pandas to compile on Android).

## Run it

```bash
cd cockpit
python3 make_cockpit.py              # one-shot generate for the live cockpit
python3 make_operator_surface.py     # one-shot generate for the operator surface
./run_local_dashboard.sh             # local server + live loop for both first-class surfaces
./run_cockpit.sh                     # live loop + optional Android browser handoff
```

`run_local_dashboard.sh` is the Android-native safe default. It regenerates every
45s, serves only on `127.0.0.1`, and never calls ADB, wireless debugging, CDP,
`am start`, or browser automation. The only first-class local pages are:

```text
http://127.0.0.1:8777/cockpit.html
http://127.0.0.1:8777/operator_surface.html
```

`run_cockpit.sh` is the convenience launcher. It also regenerates every 45s and
serves on http://127.0.0.1:8777, but may try to open Brave via Android intents
unless `COCKPIT_NO_BROWSER=1` is set.

Command deck and pilot board are no longer part of the default local loop. Keep
this stack small: live market read + operator state, nothing more theatrical.

## THE READ (what the panel tells you)

Every line is backed by a number -- not gut:

- **Who controls the tape** -- price vs VWAP
- **Where in the day's range** -- breakout / breakdown / exhaustion zones
- **Momentum** -- last 15 min, real thrust or fading
- **Volume confirmation** -- is the move backed by participation
- **Options box** -- put wall (support) <-> call wall (resistance)
- **Sentiment** -- put/call OI ratio, ATM implied vol

## Safety

Decision support only. The cockpit never places trades. You own every entry.
