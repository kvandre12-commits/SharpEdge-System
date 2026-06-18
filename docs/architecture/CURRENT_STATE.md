# SharpEdge Current State

Last updated: 2026-06-18

## Working now

- `SharpEdge-System` generates `outputs/signal.json`.
- `cockpit/trade_permission.py` computes deterministic Trade Gate scoring.
- `signal.json["trade_permission"]` contains the canonical trade gate contract.
- Phone Companion Golden Loop preserves `request_id` across:
  - request trace
  - trading view-model
  - prelaunch trace
  - launch result
  - latest observation
- Phone Companion view-model and prelaunch trace preserve a compact
  `signal_summary.trade_permission` snapshot.
- Android handoff to Brave was accepted in prior Golden Loop proof.
- `SharpEdge-Android` native Compose viewer parses and renders Trade Gate from
  `sharpedge.signal.v1` sample assets.

## Recently added viewer direction

Decision: **own the viewer**.

The primary local/mobile viewer is `SharpEdge-Android`, not Brave DevTools/CDP.
Browser launch remains useful as a debug path, but native rendering is the target.

## Install status

The current phone checkout does not have:

- `java`
- `gradle`
- Android SDK variables
- a Gradle wrapper
- a built APK

So SharpEdge Android cannot be installed directly from this Termux checkout yet.
Use Android Studio / a Gradle-capable environment / future CI artifact to build an
APK, then install that APK on the phone.

## Can the Android viewer show the system Trade Gate?

Yes, as a local packaged sample:

```bash
cd ~/SharpEdge-System
python phone_companion/export_signal_to_android_viewer.py
```

That copies `outputs/signal.json` into:

```text
~/SharpEdge-Android/app/src/main/assets/sample_signal.json
~/SharpEdge-Android/app_contracts/sharpedge.signal.v1.sample.json
```

Then rebuild/reinstall the Android app to see the current Trade Gate in the native
viewer.

Live dynamic loading from `outputs/signal.json` inside the installed Android app is
not implemented yet.

## Next highest-value steps

1. Add a Gradle wrapper or CI build for `SharpEdge-Android` APK artifacts.
2. Install the APK locally.
3. Add native viewer render observation emission.
4. Decide whether live signal sync should be file import, localhost HTTP, or share
   intent.
