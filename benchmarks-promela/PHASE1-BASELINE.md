# Phase 1 Baseline + Phase 2 Decision Log

Frozen regression baseline for the PROMELA-AGCEL integration, taken before the
action-recording rework (Phase 2). Any drift from these numbers after the
Action / `last`-cell / 44-rule wiring change is a regression.

## Baselines (frozen)

### 1. Backend load chain
- `semantics/promela-semantics.maude` (5-file chain) loads with **0 fatal errors**.
- Benign warning canary (CLI, `MAUDE_LIB=semantics`):
  - 10x "multiple distinct parses" on the decl rules (init3, decl-*).
  - 2x advisory "assignment condition fragment `# I := VIDX` bound before matching" (lines 224, 236).
  - Analysis layer (+tool2/agcel) adds 1x advisory "`_|=_` ... no common ancestor" (GlobalState overload).
  - Python runtime additionally surfaces `_,_` / `if_then_else_fi` preregularity + non-associative notes.
  - Regression signal: any NEW warning signature, or any fatal (no parse / bad token / unpatchable).

### 2. Semantics test suites
- `tests.maude` = 28, `tests-extended.maude` = 13, `tests-all.maude` = 52 (result + search), 0 fatal.

### 3. Safety benchmarks (search =>* )
| Benchmark | Safety property | Result | State space |
|---|---|---|---|
| dining-philosophers-2 | bothEating | No solution (holds) | 1473 |
| filter-2procs | bothInCS | No solution (holds) | 801 |
| qlock-2procs | bothInCS | No solution (holds) | 782 |
| onethirdrule-2procs | disagree | No solution (holds) | 749 |
| chan-buffered-1 | qOutOfOrder | No solution (holds) | 166 |
| bakery-2procs | mutexViolated | **Solution (spurious)** | ~2138 first hit |

bakery mutex is spuriously reachable under the `[loose]`-owise gap; kept as a known limitation.

### 4. RL pipeline (test.py BFS, goal reached)
| Benchmark | goal | BFS states |
|---|---|---|
| dining-philosophers-analysis | deadlock | 249 |
| filter-analysis | someInCS | 156 |
| qlock-analysis | someInCS | 112 |
| onethirdrule-analysis | bothDecided | 731 |
| chan-buffered-analysis | qDequeued | 121 |
| bakery-analysis | someInCrit | 296 |

Dining RL (300 samp / 80 eps reference): Q-table search 127, DQN search 127, Qdiff YES,
DQN success 53/80, ~2x search reduction. 500-episode figures recorded at report time.

## Phase 2 decisions (recommended defaults; ratify before wiring)

- **Q1 (record site): dedicated `last` config cell**, not a threaded top argument.
  A `<last>`-style cell is absorbed by the existing INNER-REST/CFG-REST variables, so the
  44 rules keep their term shapes; threading an argument would change every rule signature.

- **Q2 (Action ctor): `op exec : Int K -> Action`** — Int = pid of the firing process,
  K = the fired statement/label fragment. Confirmed as the recording shape.

- **Q4 (propKsize + join): default `propKsize = 1`** (keep only the head KLabel/statement of
  the K fragment) to bound the action vocabulary. Action props join `obs` by being appended
  inside `obs(S) = < obsUpto(S, <store preds> ; execProp(...)) >` — i.e. the runtime-derived
  action vocabulary is unioned with the store/network predicate vocabulary. The Python side is
  untouched; action predicates enter only through this `obs` prop list.

- **Q5 (peterson): use `filter`.** `filter-2procs` already encodes the 2-proc Peterson/filter
  (flag/turn) algorithm; do not author a separate peterson file unless a distinct N>2 filter or
  a textbook 2-flag variant is explicitly required.

- **Q6 (scope): observation-compression only, NOT action-space grouping.** Action predicates are
  added to the `obs` vocabulary; the 44-rule action space seen by MaudeEnv (`m.getRules()`) is
  left intact. Grouping the 44 rules into fewer logical actions touches MaudeEnv and is deferred.

Deferred to Phase 2 proper (gated on the above): Action sort + `last` cell + `prefix`/`propKsize`
+ 44-rule wiring (D2-D6), `execProp` / `isIndicator` focus (D3, D7), action-vs-store learning
comparison (D9). Python side (MaudeEnv, learners, AStar, train/test) stays untouched throughout.

## Update 2026-10-14: `[loose]` with the owise condition, atomic loading

The baselines above were taken while `[loose]` could fire whenever the lock was held and every
`[load-*]` rule loaded with `false`. Since then `[loose]` carries the K `[owise]` as the
condition `not canStep(...)` and `[load-*]` load with `LCK == @ PID`
(`semantics/promela-semantics.maude`). Safety verdicts are unchanged except bakery, whose
spurious violation is gone; the state spaces are smaller because a statement is no longer
interleaved between its guard and its effect.

| Benchmark | Safety property | Result | State space |
|---|---|---|---|
| dining-philosophers-2 | bothEating | No solution (holds) | 185 |
| filter-2procs | bothInCS | No solution (holds) | 336 |
| qlock-2procs | bothInCS | No solution (holds) | 255 |
| onethirdrule-2procs | disagree | No solution (holds) | 299 |
| chan-buffered-1 | qOutOfOrder | No solution (holds) | 58 |
| bakery-2procs | mutexViolated | No solution (holds) | 268 |

Test suites: `tests.maude` = 28, `tests-extended.maude` = 13, `tests-all.maude` = 52, with the
same results; `tests-atomic.maude` = 13 (new). The three spink examples marked "BLOCKED on
[loose] owise fix" (`spin-atomic1`, `spin-atomic2`, `paper-spin-atomic-handshake`) now give
No solution. Section 4 (BFS states of the RL pipeline) and the dining RL figures describe the
earlier semantics.
