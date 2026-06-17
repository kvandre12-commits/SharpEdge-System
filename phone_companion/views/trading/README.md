# Trading Views

This folder holds phone-facing trading view builders and templates.

It may read upstream artifacts such as:
- `cockpit/cockpit.html`
- `cockpit/command_deck.html`
- `outputs/operator_brief.json`
- `outputs/morning_open_dashboard.json`
- `outputs/approval_decision.json`

It must not recompute trade authority.
It renders and packages mobile-facing state only.
