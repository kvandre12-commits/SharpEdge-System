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
| `make_cockpit.py` | The live read: `cockpit.html` + `cockpit_chart.svg` + `~/SharpEdge-System/outputs/ace_snapshot.json` for SharpEdge Ace. The main event. |
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
./run_local_dashboard.sh             # local server + live loop for first-class surfaces
# COCKPIT_OPEN_BROWSER=1 COCKPIT_OPEN_OPERATOR_SURFACE=1 ./run_local_dashboard.sh
# COCKPIT_INTERVAL=10 ./run_local_dashboard.sh  # 10s loop
./run_local_ace_dashboard.sh         # same shell, but Ace-style authority for execution logic
./run_cockpit.sh                     # live loop + optional Android browser handoff
./open_cockpit.sh                    # re-open cockpit fast if you closed the tab like a maniac
./remind_open_cockpit.sh             # Android tap-to-open reminder notification
./run_runner_handoff_feature.sh      # dedicated root-port handoff feature page on 8765
```

`run_local_dashboard.sh` is the Android-native safe default. It regenerates every
`COCKPIT_INTERVAL` seconds (`5` by default; use `10` for the slower shared loop),
serves only on `127.0.0.1`, refreshes the operator artifact chain before
rendering the operator surface, writes `outputs/ace_snapshot.json` on the same
heartbeat for SharpEdge Ace, and never calls ADB, wireless debugging, CDP,
`am start`, or browser automation. The only first-class local pages are:

```text
http://127.0.0.1:8777/cockpit.html
http://127.0.0.1:8777/operator_surface.html
http://127.0.0.1:8777/runner_handoff_live.html
http://127.0.0.1:8777/regime_nerv_split.html
```

`regime_nerv_split.html` places the normal live cockpit next to a Regime/NERV
sidecar panel. The sidecar reads current disposable desk artifacts when present:
`outputs/nerv_trade_desk/ctc_nerv_trade_desk.json` and
`outputs/regime_cartridges/*/desk/ctc_nerv_trade_desk.json`. If runtime cleanup
has removed those boards, it shows an honest empty-state instead of pretending.

Those are the live surfaces. Treat preview/mock artifacts like `handoff_preview.html`
as demos only, not the always-live operator view. `runner_handoff_live.html`
uses the live signal too, but stays honest: it only goes full handoff-promotion
when a real `EXHAUSTION -> RUNNER HANDOFF` is active.

If you want the old dedicated-feature feel, use:

```text
http://127.0.0.1:8765/cockpit/runner_handoff_live.html
```

via `./run_runner_handoff_feature.sh`, which serves the repo root on its own
port so the handoff page is not competing with the regular cockpit dashboard.

`run_cockpit.sh` is the convenience launcher. It also regenerates every 45s and
serves on http://127.0.0.1:8777, but may try to open Brave via Android intents
unless `COCKPIT_NO_BROWSER=1` is set. If you want a little shove, run it with
`COCKPIT_REMIND=1` to also post a tap-to-open Android reminder notification.

If you just need the cockpit back in your face later, use:

```bash
./open_cockpit.sh
./remind_open_cockpit.sh
```

The first one re-opens the page immediately. The second sends a notification you
can tap later like a civilized goblin.

If you want the cockpit shell but the trimmed Ace-style authority lane, use:

```bash
./run_local_ace_dashboard.sh
```

That keeps the graphs, carry maps, day-bucket context, and event receipts while
reducing the permission contract to the core execution spine.

For unattended Android starts at the New York market open, use:

```bash
bash ../ops/run_ace_authority_market_open_daemon.sh
bash ../ops/install_termux_boot_ace_authority.sh
```

The daemon watches New York time and launches the Ace-authority dashboard once
per weekday at `09:30`.

If you want the regular live cockpit to be open and ready in Brave before the
trading day, use:

```bash
bash ../ops/run_cockpit_ready_daemon.sh
bash ../ops/install_termux_boot_cockpit_ready.sh
```

That scheduler watches New York time and, once per weekday at `09:00` by
default, will either:
- launch the live localhost dashboard and open Brave to `cockpit.html`, or
- if the dashboard is already running, simply re-open the cockpit in Brave.

Useful env knobs:

```bash
SHARPEDGE_READY_HHMM=0845 bash ../ops/run_cockpit_ready_daemon.sh
SHARPEDGE_READY_TZ=America/New_York bash ../ops/install_termux_boot_cockpit_ready.sh
```

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
