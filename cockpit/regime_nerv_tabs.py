"""Accessible tab shell for SharpEdge operator surfaces."""

from __future__ import annotations


def render_tabs_html(*, refresh_seconds: int = 5) -> str:
    """Render stable tabs; child surfaces own their individual refresh cycles."""
    del refresh_seconds  # Compatibility: parent refresh is intentionally disabled.
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SharpEdge Operator Tabs</title>
  <style>
    html, body { margin:0; height:100%; background:#05070b; color:#e8eefc; font-family:system-ui, sans-serif; }
    .bar { min-height:46px; display:flex; align-items:center; gap:8px; padding:6px 10px; box-sizing:border-box; background:#07101e; border-bottom:1px solid #20304d; flex-wrap:wrap; }
    .tablist { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
    button, a { border:1px solid #2c466d; background:#111f36; color:#c9d8f3; border-radius:999px; padding:7px 12px; font:13px system-ui; text-decoration:none; }
    button.active, button[aria-selected="true"] { background:#1f6feb; color:white; border-color:#58a6ff; }
    button:focus-visible, a:focus-visible { outline:3px solid #f8d36f; outline-offset:2px; }
    .links { margin-left:auto; display:flex; gap:8px; flex-wrap:wrap; }
    iframe { display:none; border:0; width:100%; height:calc(100vh - 58px); background:#08111f; }
    iframe.active { display:block; }
    @media (max-width: 900px) { .links { margin-left:0; } iframe { height:calc(100vh - 96px); } }
  </style>
</head>
<body>
  <nav class="bar" aria-label="SharpEdge surfaces">
    <div class="tablist" role="tablist" aria-label="Operator views">
      <button id="decision-tab" type="button" role="tab" aria-selected="false" aria-controls="decision-panel" tabindex="-1" onclick="showTab('decision')">Decision</button>
      <button id="graph-tab" type="button" role="tab" aria-selected="false" aria-controls="graph-panel" tabindex="-1" onclick="showTab('graph')">Graph + Read</button>
      <button id="spine-tab" type="button" role="tab" aria-selected="false" aria-controls="spine-panel" tabindex="-1" onclick="showTab('spine')">Spine</button>
      <button id="options-tab" type="button" role="tab" aria-selected="false" aria-controls="options-panel" tabindex="-1" onclick="showTab('options')">Options</button>
    </div>
    <div class="links">
      <a href="hey_guy.html">hey guy</a>
      <a href="regime_nerv_split.html">split</a>
    </div>
  </nav>
  <main>
    <iframe id="decision-panel" role="tabpanel" aria-labelledby="decision-tab" src="operator_decision_card.html" title="SharpEdge decision card" hidden></iframe>
    <iframe id="graph-panel" role="tabpanel" aria-labelledby="graph-tab" src="cockpit.html" title="SharpEdge graph and read" hidden></iframe>
    <iframe id="spine-panel" role="tabpanel" aria-labelledby="spine-tab" src="operator_surface.html" title="SharpEdge spine surface" hidden></iframe>
    <iframe id="options-panel" role="tabpanel" aria-labelledby="options-tab" src="regime_nerv_panel.html" title="SharpEdge options and NERV panel" hidden></iframe>
  </main>
  <script>
    const KEY = 'sharpedge.activeTab';
    const TABS = ['decision', 'graph', 'spine', 'options'];
    function showTab(name, moveFocus = false) {
      const selectedName = TABS.includes(name) ? name : 'decision';
      localStorage.setItem(KEY, selectedName);
      for (const tab of TABS) {
        const selected = tab === selectedName;
        const button = document.getElementById(tab + '-tab');
        const panel = document.getElementById(tab + '-panel');
        button.classList.toggle('active', selected);
        button.setAttribute('aria-selected', String(selected));
        button.tabIndex = selected ? 0 : -1;
        panel.classList.toggle('active', selected);
        panel.hidden = !selected;
        if (selected && moveFocus) button.focus();
      }
    }
    document.querySelector('[role="tablist"]').addEventListener('keydown', (event) => {
      const current = TABS.findIndex((tab) => document.getElementById(tab + '-tab') === document.activeElement);
      if (current < 0) return;
      let next = current;
      if (event.key === 'ArrowRight') next = (current + 1) % TABS.length;
      else if (event.key === 'ArrowLeft') next = (current - 1 + TABS.length) % TABS.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = TABS.length - 1;
      else return;
      event.preventDefault();
      showTab(TABS[next], true);
    });
    showTab(localStorage.getItem(KEY));
  </script>
</body>
</html>
"""
