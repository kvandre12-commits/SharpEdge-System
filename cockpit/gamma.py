"""0DTE gamma read for the SharpEdge cockpit.

Answers the question that defines a 0DTE day: is the tape PINNED or LOOSE?

  - Positive net dealer gamma  -> dealers dampen moves -> price PINS toward a
    high-gamma strike -> chop / mean-reversion -> fade the edges.
  - Negative net dealer gamma  -> dealers amplify moves -> price runs ->
    trend / "wheee" -> ride momentum, directional 0DTE.

Computed from the CBOE chain (gamma + open_interest per strike). Convention:
market makers assumed long calls / short puts (standard GEX convention), so
net gamma = sum(callGamma*callOI) - sum(putGamma*putOI). Sign is what matters.

Also computes:
  - gamma pin  : strike with the largest total gamma*OI (the magnet)
  - max pain   : strike minimizing total ITM option payout at expiry
"""

from __future__ import annotations

import datetime as dt


def _nearest_expiry(book):
    today = dt.date.today()
    future = sorted(e for e in book if e >= today)
    return (future or sorted(book))[0]


def max_pain(chain, strikes):
    """Strike where the fewest option dollars finish in-the-money."""
    best_k, best_pay = None, float("inf")
    for K in strikes:
        pay = 0.0
        for s in strikes:
            coi = chain[s].get("C", {}).get("open_interest", 0) or 0
            poi = chain[s].get("P", {}).get("open_interest", 0) or 0
            if s < K:
                pay += (K - s) * coi   # calls in the money
            elif s > K:
                pay += (s - K) * poi   # puts in the money
        if pay < best_pay:
            best_pay, best_k = pay, K
    return best_k


def gamma_profile(book, spot):
    """Return the 0DTE gamma read as a dict (fail-soft -> {})."""
    if not book:
        return {}
    exp = _nearest_expiry(book)
    chain = book[exp]
    strikes = sorted(chain.keys())
    if not strikes:
        return {}

    net_gamma = 0.0
    pin_strike, pin_val = None, -1.0
    for k in strikes:
        c = chain[k].get("C", {})
        p = chain[k].get("P", {})
        cg = (c.get("gamma", 0) or 0) * (c.get("open_interest", 0) or 0)
        pg = (p.get("gamma", 0) or 0) * (p.get("open_interest", 0) or 0)
        net_gamma += cg - pg
        magnet = cg + pg          # total gamma concentration at this strike
        if magnet > pin_val:
            pin_val, pin_strike = magnet, k

    mp = max_pain(chain, strikes)
    dte = (exp - dt.date.today()).days
    pin_dist = (pin_strike - spot) / spot * 100 if pin_strike else 0.0

    return {
        "exp": exp.isoformat(),
        "dte": dte,
        "regime": "positive" if net_gamma >= 0 else "negative",
        "net_gamma": net_gamma,
        "pin": pin_strike,
        "pin_dist": pin_dist,
        "max_pain": mp,
        "spot": spot,
    }


def gamma_card(gp):
    """Render the gamma profile as a cockpit card dict, or None."""
    if not gp:
        return None
    pin = gp["pin"]
    mp = gp["max_pain"]
    spot = gp["spot"]
    dte_lbl = "0DTE" if gp["dte"] == 0 else f"{gp['dte']}DTE"

    if gp["regime"] == "positive":
        kind = "info"
        # which way the magnet pulls price
        if pin and spot > pin:
            pull = f"magnet ${pin:g} is BELOW price - expect pull DOWN"
        elif pin and spot < pin:
            pull = f"magnet ${pin:g} is ABOVE price - expect pull UP"
        else:
            pull = f"price stuck on magnet ${pin:g}"
        tag = "STICKY DAY (calm/chop)"
        bias = "FADE the edges - bet on snap-back to the magnet"
        detail = (f"{dte_lbl} {gp['exp']} | {pull} | "
                  f"careful: cheap 0DTE lottos usually bleed today "
                  f"(tech: positive gamma, max pain ${mp:g})")
    else:
        kind = "warn"
        tag = "RUNNER DAY (wheee)"
        bias = "RIDE momentum - go directional, breakouts run"
        detail = (f"{dte_lbl} {gp['exp']} | no strong magnet holding price, "
                  f"moves snowball | good day for directional 0DTE "
                  f"(tech: negative gamma, max pain ${mp:g})")

    return {"tag": tag, "bias": bias, "kind": kind, "detail": detail}
