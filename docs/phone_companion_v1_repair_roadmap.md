# Phone Companion v1 Repair Roadmap

## Mission

The mission is:

```text
Intent → Contract → Code Puppy → DroidPuppy → Phone Action → Observation
```

Every recommendation in this roadmap is judged against one question:

**Does this increase the probability that Phone Companion v1 completes that loop reliably, with minimal confusion, for a tired future pilot?**

If not, it does not belong in v1.

---

## Architectural posture

Assume the following are true:

1. future pilots will lose context
2. future pilots will make bad decisions when tired
3. architecture complexity is a liability
4. every new subsystem must justify its existence

That means Phone Companion v1 should optimize for:

- one thin operational path
- explicit contracts
- obvious ownership boundaries
- deterministic launch behavior
- durable observations
- boring recovery steps

Not for cleverness.
Not for breadth.
Not for “platform” fantasies.

---

## Current architectural truth

From `current_state_report.md` and `phone_companion_v1_pilot_recovery_design_doc.md`, the situation is this:

- the **docs are stronger than the runtime**
- the `phone_companion/` folder is mostly **scaffold + contracts + stubs**
- the **operator/safety artifact chain upstream is the healthiest subsystem**
- `ops/section_gap/` is **not yet a runnable lane**
- backend/frontend surfaces are **not trustworthy** and should not become accidental dependencies
- the repo already knows the correct execution spine, but Phone Companion does **not** yet complete it

So the roadmap must do one thing well:

**turn Phone Companion from architecture paperwork into one thin, survivable operational loop.**

---

## Decision filter for all work

Before any task is accepted, ask:

1. Does it help one of the six mission layers?
2. Does it reduce ambiguity for a future pilot?
3. Does it avoid creating a new subsystem?
4. Can it be explained in one paragraph to a tired operator?
5. Would v1 still function if this task were delayed?

If the answer to #1 or #2 is no, reject it.
If the answer to #3 is no, be suspicious.
If the answer to #4 is no, simplify it.
If the answer to #5 is yes, it probably does not belong in NOW.

---

# NOW

The NOW section is intentionally short.
If everything is urgent, nothing is urgent.

## 1. Freeze v1 scope to one operational lane: trading only

**Objective**  
Make Phone Companion v1 prove the loop using the trading lane only. Formally defer section-gap runtime work until after one production-like path works end-to-end.

**Why it matters**  
The trading lane already has upstream artifacts, existing cockpit patterns, and a clearer operator surface. Section-gap currently has architecture, not runtime truth. If both lanes advance together, Phone Companion will become a vague shell for two half-systems instead of one working system.

**Risk if ignored**  
- split attention
- launcher behavior that tries to generalize too early
- fake progress through parallel stubs
- future pilots touching section-gap before the shell itself is proven

**Effort estimate**  
Small

**Dependency chain**  
None

**Success criteria**  
- v1 runtime scope is explicitly documented as **trading-first**
- section-gap is marked **deferred for runtime implementation**
- no new Phone Companion runtime work begins in section-gap until trading completes the full loop

---

## 2. Stabilize the baseline and make the Phone Companion docs/scaffold a committed starting point

**Objective**  
Turn the current local architecture package into a stable baseline that future pilots can actually start from.

**Why it matters**  
Right now too much of the Phone Companion context lives in a dirty working tree. That is how context gets lost, duplicated, or contradicted. A future pilot cannot recover from architecture that only half-exists.

**Risk if ignored**  
- context loss after device reset or branch drift
- duplicate docs saying slightly different things
- pilots rebuilding the same folder structure from memory
- “which file is authoritative?” confusion

**Effort estimate**  
Small

**Dependency chain**  
Task 1 helps, but is not strictly required

**Success criteria**  
- `phone_companion/` scaffold and core docs are committed as a coherent baseline
- `docs/phone_companion_v1_pilot_recovery_design_doc.md` is explicitly referenced as the first recovery document
- the roadmap itself is part of the committed baseline
- a future pilot can clone the repo and find the Phone Companion home without guessing

---

## 3. Define and validate the minimum contract chain for the trading path

**Objective**  
Create the smallest complete contract path for trading:

- one request example
- one view-model example
- one observation example
- one validation/test path that proves their required fields and relationships

**Why it matters**  
The contract seam is the survival seam. If request, view-model, and observation drift, Phone Companion becomes a pile of ad hoc shell logic. Tired pilots will absolutely take shortcuts unless the seam is explicit and testable.

**Risk if ignored**  
- launcher scripts become business logic dumpsters
- field drift across layers
- observations that cannot be correlated to requests
- future rewrites because nobody trusts the contract shape

**Effort estimate**  
Small to Medium

**Dependency chain**  
Task 1, Task 2

**Success criteria**  
- one golden trading request fixture exists
- one golden trading view-model fixture exists
- one golden observation fixture exists
- validation/tests fail on missing required fields or broken linkage (`request_id`, `view_id`, timestamps, target, status)
- future pilots can inspect examples instead of reverse-engineering intent from shell scripts

---

## 4. Complete one thin operational loop for trading: launch + observe

**Objective**  
Implement the minimum runtime that takes a valid trading request and completes:

```text
Intent → Contract → Code Puppy → DroidPuppy → Phone Action → Observation
```

for one concrete action:

- open the trading dashboard in Brave
- write an observation record after the attempt

**Why it matters**  
Without this, Phone Companion is still architecture cosplay. The system does not need multiple views, clever routing, or backend/frontend complexity. It needs one repeatable loop.

**Risk if ignored**  
- endless planning without operational proof
- future pilots adding extra abstractions to avoid confronting the missing loop
- no evidence that the shell can survive real device interaction

**Effort estimate**  
Medium

**Dependency chain**  
Task 1 → Task 2 → Task 3

**Success criteria**  
- one trading launcher accepts a known request shape
- launcher verifies required upstream artifacts before launch
- launcher invokes DroidPuppy through the preferred path for Brave open
- launcher writes a `phone_companion_observation_v1` record on success/failure
- the observation includes `request_id`, action target, start/end timestamps, status, and result summary
- a future pilot can run one command and see both the phone action and the recorded observation

---

## 5. Write the operator recovery runbook and preflight checklist

**Objective**  
Create a brutally simple recovery-and-run document for Phone Companion runtime use.

It should answer:
- what to read first
- what artifacts must exist
- what command to run
- what “success” looks like
- what to inspect when it fails

**Why it matters**  
Architecture dies when only the original builder knows the ritual. The runbook is the anti-ritual. It converts tribal memory into procedure.

**Risk if ignored**  
- pilots skipping required preflight checks
- repeated breakage from stale or missing artifacts
- “it didn’t work” with no shared troubleshooting path
- context loss after a week away from the repo

**Effort estimate**  
Small

**Dependency chain**  
Task 2, Task 3, Task 4

**Success criteria**  
- one runbook exists under `docs/`
- it includes preflight, run, expected outputs, and failure triage
- a tired pilot can follow it without needing architecture interpretation skills
- it references the exact source docs in the right order

---

# NEXT

These are important, but they only matter after the trading loop exists.

## 6. Add explicit launcher failure semantics and fallback order

**Objective**  
Make launcher behavior deterministic about how it tries delivery:

1. direct handoff / intent path
2. Brave URL open
3. structured share path if relevant
4. UI fallback only as last resort

**Why it matters**  
Future pilots will otherwise add fallback behavior randomly. That creates spooky-action launch behavior nobody can predict.

**Risk if ignored**  
- brittle UI automation becoming the default
- no shared meaning for “launch failed”
- repeated manual debugging of the same device edge cases

**Effort estimate**  
Medium

**Dependency chain**  
Task 4, Task 5

**Success criteria**  
- fallback order is documented and enforced
- each failure path writes an observation with clear status
- UI fallback is visibly exceptional, not normal runtime behavior

---

## 7. Add authority and freshness preflight gates to Phone Companion entrypoints

**Objective**  
Ensure Phone Companion explicitly checks the upstream artifact state it depends on before launch.

For trading, that means at minimum consuming:
- `approval_decision.json`
- `workflow_state.json`
- required display artifacts

**Why it matters**  
Phone Companion must not become a permission engine, but it also must not behave as if stale or blocked upstream state does not exist.

**Risk if ignored**  
- the shell shows comforting views disconnected from current reality
- operators treat stale dashboards as fresh state
- future code quietly bypasses upstream blockers

**Effort estimate**  
Medium

**Dependency chain**  
Task 3, Task 4

**Success criteria**  
- launcher preflight reports missing/stale/blocked state explicitly
- read-only viewing behavior vs blocked action behavior is documented
- Phone Companion never overrides upstream authority

---

## 8. Add a tiny test harness for Phone Companion itself

**Objective**  
Create targeted tests for the Phone Companion shell layer only.

Focus on:
- contract validation
- artifact presence checks
- observation writing
- launcher decision behavior in dry-run mode

**Why it matters**  
The shell is currently under-documented in code and over-documented in prose. A few sharp tests will preserve behavior better than another speech about architecture.

**Risk if ignored**  
- regressions in the exact layer most likely to be “quick-fixed” by tired pilots
- shell logic drifting away from contracts and docs
- fake confidence because the docs still sound clean

**Effort estimate**  
Medium

**Dependency chain**  
Task 3, Task 4, Task 7

**Success criteria**  
- targeted Phone Companion tests exist
- dry-run launcher behavior is testable without real device dependency
- observation outputs are assertable and stable

---

## 9. Define the entry criteria for section-gap before writing its runtime

**Objective**  
Document the minimum upstream readiness that section-gap must meet before it is allowed to use the Phone Companion shell.

**Why it matters**  
Section-gap is the easiest place for ambition to outrun evidence. It should not ride along on the trading lane’s shell until it has real inputs to consume.

**Risk if ignored**  
- shell complexity added to support a domain that is still hypothetical
- duplicated abstractions for events/observations that do not exist yet
- future pilots building UI around empty folders

**Effort estimate**  
Small

**Dependency chain**  
Task 1, Task 5

**Success criteria**  
- section-gap has a written runtime-entry checklist
- checklist names required artifacts and non-goals
- no runtime work begins before those entry criteria are met

---

# LATER

These are legitimate only after the shell is proven and boring.

## 10. Add section-gap as a second lane only after upstream domain artifacts exist

**Objective**  
Use the proven Phone Companion shell pattern for section-gap once section-gap has real event, observation, and dashboard artifacts.

**Why it matters**  
Reusing the shell pattern is good. Sharing half-baked runtime assumptions is not.

**Risk if ignored**  
If done too early, the shell becomes abstract mush. If done later and properly, the second lane becomes evidence that the architecture generalizes.

**Effort estimate**  
Large

**Dependency chain**  
Task 4, Task 7, Task 8, Task 9, plus section-gap upstream artifact readiness

**Success criteria**  
- section-gap runtime uses the same mission spine without changing its meaning
- no trading semantics leak into section-gap files
- lane-specific view logic remains separate

---

## 11. Add support-bundle style troubleshooting for Phone Companion failures

**Objective**  
Package failure evidence cleanly when device launch or handoff fails.

**Why it matters**  
Once the shell is operational, the next survivability issue will be debugging real Android/device behavior.

**Risk if ignored**  
- repeated unstructured debugging
- hard-to-reproduce failures
- no durable evidence trail for device issues

**Effort estimate**  
Medium

**Dependency chain**  
Task 4, Task 6, Task 8

**Success criteria**  
- failure observations can point to support artifacts or debug context
- device/runtime failures become diagnosable instead of mystical

---

## 12. Re-evaluate whether a dedicated backend/frontend is needed at all

**Objective**  
Only introduce a maintained app surface if the thin shell proves insufficient.

**Why it matters**  
The current backend/frontend state is a warning label, not an invitation. Phone Companion v1 should not inherit dead-weight architecture just because a folder exists.

**Risk if ignored**  
- new subsystem created out of habit, not necessity
- duplicated state and new failure modes
- Phone Companion becoming a mini-platform instead of a shell

**Effort estimate**  
Large

**Dependency chain**  
Task 4 through Task 11, plus clear evidence that shell-only operation is insufficient

**Success criteria**  
- there is a written justification for any backend/frontend addition
- the new subsystem has a narrower job than the shell, not a broader one
- the mission spine stays intact

---

# NEVER

These are anti-goals. Do not “get creative” here.

## 13. Never let Phone Companion become a domain brain

**Objective**  
Preserve the rule that Phone Companion reads domain truth but does not mint it.

**Why it matters**  
The moment Phone Companion starts recomputing trading truth or section-gap classification, it stops being a shell and becomes an untestable second brain.

**Risk if ignored**  
- conflicting truth between upstream systems and the shell
- duplicated logic
- invisible authority drift

**Effort estimate**  
Zero to preserve, very expensive to fix later

**Dependency chain**  
Applies always

**Success criteria**  
- Phone Companion consumes artifacts; it does not redefine them
- no lane logic is duplicated inside launchers or observation code

---

## 14. Never build a generic Phone Companion platform in v1

**Objective**  
Refuse the urge to create a universal mobile orchestration framework before one use case works.

**Why it matters**  
Generic platforms are where focused projects go to die with excellent folder names.

**Risk if ignored**  
- abstraction before evidence
- too many configuration layers
- nobody knows the real runtime path anymore

**Effort estimate**  
Zero to avoid, catastrophic if indulged

**Dependency chain**  
Applies always

**Success criteria**  
- v1 remains a thin shell with one proven lane first
- any added generality follows actual repeated use, not imagination

---

## 15. Never make UI fallback the default path

**Objective**  
Keep UI automation as the emergency tool, not the architecture.

**Why it matters**  
UI steering is fragile, device-specific, and hard to reason about under fatigue.

**Risk if ignored**  
- brittle launches
- hard-to-debug regressions
- fake success because “a tap happened somewhere”

**Effort estimate**  
Zero to avoid

**Dependency chain**  
Applies always

**Success criteria**  
- direct handoff and explicit launch paths remain primary
- any UI fallback is recorded as fallback in observations

---

## 16. Never let Phone Companion override approval authority

**Objective**  
Protect the rule that `approval_decision` wins.

**Why it matters**  
A phone shell that escalates authority is not a shell. It is a safety breach wearing a nice mobile face.

**Risk if ignored**  
- silent permission drift
- false operator trust
- unsafe behavior under stale or partial data

**Effort estimate**  
Zero to preserve

**Dependency chain**  
Applies always

**Success criteria**  
- launchers and views display authority state; they do not rewrite it
- downstream shell behavior stays subordinate to upstream approval

---

# If I only had one weekend, this is exactly what I would do

1. **Freeze runtime scope to trading only.**  
   No section-gap runtime work. No backend/frontend rescue mission. No extra lanes.

2. **Commit the Phone Companion baseline.**  
   Make the current docs and folder scaffold a clean, recoverable starting point.

3. **Create one golden trading request, one golden view-model, and one golden observation example.**  
   Add the smallest validation/tests needed so the seam is real.

4. **Implement one thin trading launcher that does exactly one thing:**  
   verify trading artifacts, open the chosen dashboard in Brave through DroidPuppy, and write an observation record.

5. **Write one runbook for tired humans.**  
   Preflight, run command, expected output, failure triage.

That is the weekend plan because it produces the first thing Phone Companion v1 currently lacks:

**one survivable operational loop that a future pilot can actually rerun.**
