# Execution Spine Vertical Contracts

Status: active design record / constitution for execution-spine cleanup.

Purpose: make each vertical explicit and comparable.
If a future change does not fit the contract for a vertical, it should be:
- rejected,
- re-homed,
- or promoted into a new vertical only after earning a coherent contract.

Common pattern for every vertical:

Input Boundary
-> State Engine
-> Explicit State Packet
-> Derived Score

Common audit questions:
1. Who owns this?
2. What are its inputs?
3. Who owns the candidate set?
4. What state packet does it produce?
5. How is the legacy score derived from that state?

---

## Structure

- vertical: Structure
- owns:
  - price sequence
  - confirmed swing progression
  - HH/HL, LH/LL, mixed, insufficient sequence
- does_not_own:
  - failed breaks
  - reclaims
  - acceptance
  - compression
  - momentum
- candidate_owner:
  - bars / swing extraction
- inputs:
  - bars
- outputs:
  - structure_state_packet
- derived:
  - structure_score

## Acceptance

- vertical: Acceptance
- owns:
  - level interaction via repeated close-based acceptance
  - accepted_above / accepted_below / no_acceptance / insufficient_data
- does_not_own:
  - trend
  - VWAP drift fallback
  - failed breaks
  - wall importance
  - choosing which levels matter
- candidate_owner:
  - shared_reference_map candidate list for acceptance
- inputs:
  - candidate levels
  - recent closes
- outputs:
  - acceptance_state_packet
- derived:
  - acceptance_score

## Location

- vertical: Location
- owns:
  - spatial relationship to references
  - nearest reference
  - above / below / near / between references
  - distance to relevant references
- does_not_own:
  - balance
  - stretch
  - edge
- candidate_owner:
  - shared_reference_map
- inputs:
  - references
  - current_price
- outputs:
  - location_state_packet
- derived:
  - location_score

## Dealer

- vertical: Dealer
- owns:
  - dealer microstructure
  - gamma regime
  - pin gravity
  - wall pressure
  - dealer-driven expansion / damping context
- does_not_own:
  - premium richness outside dealer interpretation
  - price trend
  - execution permission
  - location
- candidate_owner:
  - options context feed
- inputs:
  - gamma profile
  - wall data
  - spot
  - options-derived dealer context
- outputs:
  - dealer_state_packet
- derived:
  - dealer_score

## Volume

- vertical: Volume
- owns:
  - participation confirmation
  - whether volume is confirming, participating, mixed, or missing
- does_not_own:
  - trend
  - price direction by itself
  - broader tape energy outside participation confirmation
- candidate_owner:
  - volume feature builder
- inputs:
  - volume features
  - recent bar path for participation alignment
- outputs:
  - volume_state_packet (`sharpedge.volume_profile.v1`)
- derived:
  - volume_score

## Trend

- vertical: Trend
- owns:
  - trend-component alignment
  - short-horizon path
  - light VWAP relationship context
  - momentum agreement / disagreement
- does_not_own:
  - structure
  - acceptance
  - volume confirmation
  - dealer microstructure
  - time-of-day context
  - standalone direction prediction in isolation
- candidate_owner:
  - trend context / price-action feature builder
- inputs:
  - bars
  - vwap relationship
  - momentum features
- outputs:
  - trend_state_packet (`sharpedge.trend_state.v1`)
  - compact state family:
    - aligned_up
    - aligned_down
    - conflict
    - neutral
    - insufficient
- reason layer:
  - reasons like `vwap_chop` and `vwap_rotation` explain the state
  - they are not top-level states themselves
- derived:
  - trend_score

## Time

- vertical: Time
- owns:
  - session timing context
  - regular-session window classification
- does_not_own:
  - trend
  - balance
  - volatility
  - setup identity
  - directional prediction
- candidate_owner:
  - market session clock / session context
- inputs:
  - session timestamp
  - market session schedule
- outputs:
  - time_state_packet (`sharpedge.time_state.v1`)
  - compact state family:
    - opening
    - morning
    - midday
    - afternoon
    - power_hour
    - closed_or_unknown
- reason layer:
  - reasons like `opening_auction`, `midday_chop`, and `afternoon_rotation`
  - explain the window; they do not replace the state itself
- derived:
  - time_score

---

## Quarantine rule

If a concept is not a domain fact for one vertical, quarantine it instead of smuggling it into the nearest score.

Examples of likely quarantine/composite concepts:
- balance
- stretch
- edge
- broader auction behavior

Those concepts may later earn their own verticals, but only if they can answer the same contract fields:
- owns
- does_not_own
- candidate_owner
- inputs
- outputs
- derived

---

## Current doctrinal notes

- Structure and Acceptance are already moving to this state-first contract.
- Location should be purified toward pure spatial facts.
- Dealer should become one explicit packet with subfields rather than one blended heuristic blob.
- Volume already has the closest thing to a real state packet in `sharpedge.volume_profile.v1`.
- Trend and Time still need explicit state packets to complete the pattern.

---

## Usage rule

When proposing a new feature, first ask:
- which vertical owns this?
- who owns its candidate set?
- is it a domain fact or a cross-domain interpretation?

If it is cross-domain, do not jam it into an existing state engine just because the code is nearby.
