# Launchers

Launchers are thin wrappers that:
- ensure required artifacts already exist
- optionally start a local static server
- open Brave or invoke share/handoff actions
- emit observations on success/failure

They must stay thin.
They do not own domain truth.

Useful entrypoints:
- `run_phone_companion_trading.sh` — open cockpit dashboard flow
- `run_phone_companion_android_signal_import.sh` — export latest signal and hand it to SharpEdge-Android via direct intent import
- `run_phone_companion_android_operator_import.sh` — export latest operator packet and hand it to SharpEdge-Android via direct intent import
- `run_phone_companion_android_live.sh` — rebuild cockpit + operator artifacts, then import the fresh operator packet straight into SharpEdge-Android
