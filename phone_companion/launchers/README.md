# Launchers

Launchers are thin wrappers that:
- ensure required artifacts already exist
- optionally start a local static server
- open Brave or invoke share/handoff actions
- emit observations on success/failure

They must stay thin.
They do not own domain truth.
