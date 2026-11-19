# SharpEdge Cockpit Reboot Continuity

The live cockpit process is not durable by itself. Android, a phone reboot, or a
Termux process reset can stop it. SharpEdge therefore uses an idempotent Android
JobScheduler watchdog as the primary reboot-continuity mechanism.

## Components

- `ops/run_cockpit_watchdog_once.sh` checks `127.0.0.1:8777` and exits when the
  station is healthy. If it is down, the watchdog acquires an atomic lock,
  rechecks health, obtains a Termux wake lock, starts the dashboard headlessly,
  and verifies the port.
- `ops/install_termux_job_cockpit_watchdog.sh` installs Android job ID `8777`
  every 15 minutes with `persisted=true`.
- `~/.termux/boot/start_sharpedge_cockpit_ready_scheduler.sh` remains a secondary
  fallback if the separate Termux:Boot application is installed later.

The watchdog never changes trading authority. It only restores the local
cockpit/operator process. Robinhood bindings and approval gates remain separate.

## Install or repair

```bash
cd ~/SharpEdge-System
bash ops/install_termux_job_cockpit_watchdog.sh
```

Expected receipt includes:

```text
Pending Job 8777
periodic: 900000ms
persisted
```

## Verify

```bash
termux-job-scheduler --pending
curl -fsS http://127.0.0.1:8777/cockpit.html >/dev/null
tail -n 20 outputs/cockpit_watchdog.log
```

The durable health/restart receipts are in:

- `outputs/cockpit_watchdog.log`
- `outputs/cockpit_watchdog_dashboard.log`
- `outputs/cockpit_watchdog_dashboard.pid`

## Run once manually

```bash
bash ops/run_cockpit_watchdog_once.sh
```

This is safe to repeat. A healthy station produces `action=none`; concurrent
restart attempts are suppressed by `outputs/.cockpit_watchdog.lock`.

## Roll back

```bash
termux-job-scheduler --cancel --job-id 8777
```

Canceling the job does not stop an already-running cockpit process. Stop that
process separately if desired.

## Honest limitations

- Android JobScheduler is approximate, not exact-time scheduling. The minimum
  periodic interval is 15 minutes.
- `persisted=true` requests reboot survival from Android. A real reboot test is
  the only conclusive proof on a specific phone/ROM.
- OEM battery restrictions or force-stopping Termux/Termux:API can suppress jobs.
  Exempt both applications from aggressive battery optimization when continuous
  operation matters.
- This protects against reboot/process loss, not deletion of Termux app data or
  loss of the phone. Cross-device disaster recovery additionally requires the
  source changes to be committed and pushed to a trusted remote.
