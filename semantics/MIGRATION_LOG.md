# MIGRATION_LOG.md

Task 3 backend restructuring: removed sorts/operators and their replacements.

## Module reorganization (Step 0)

The four files keep their names but the modules inside are reorganized to match the five-category separation:

- **k-prelude.maude** — sort/prelude + syntax categories. Modules: `PROMELA-VALUE`, `PROMELA-NAMES`, `PROMELA-LOC-REF`, `PROMELA-LOCK`, `PROMELA-MSG`, `PROMELA-QUEUE`, `PROMELA-K`, `PROMELA-GOTOCTX`, views (Id, LVal, Ref, Value, Queue, GotoCtx), `PROMELA-SYNTAX`, `PROMELA-EFFECTS`, `PROMELA-BRANCH`.
- **promela-state.maude** — configuration category. Modules: `PROMELA-K-CONFIG`, `PROMELA-LOCAL-SET`, `PROMELA-MAPS` (central parametric MAP/LIST instantiations), `PROMELA-PROC-CELL`, `PROMELA-PROCTYPE-CELL`, `PROMELA-CONFIG`.
- **promela-functions.maude** — functions category. Modules unchanged in role: `PROMELA-EVAL`, `PROMELA-PURIFY`, `PROMELA-EXEC`, `PROMELA-EFFECT`, `PROMELA-LOAD`, `PROMELA-HCHAN`.
- **promela-semantics.maude** — rules category. Module: `PROMELA-SEMANTICS`.

## Removed sorts/operators and their replacements

| Removed | Replacement |
|---|---|
| `sort State` | `sort KConfig` (AC multiset of `KConfigItem`) |
| `op state : Proctypes Procs Store Network Lock List{Int} Int Int Int Int -> State [ctor]` | The configuration is the multiset itself; no wrapper. Cells juxtaposed via `__`. |
| `sort Env` | `Map{LVal, Ref}` (parametric prelude MAP) |
| `sort Store` | `Map{Nat, Value}` (parametric prelude MAP) |
| `sort Network` | `Map{Nat, Queue}` (parametric prelude MAP) |
| `sort GotoMap` | `Map{Id, GotoCtx}` (parametric prelude MAP) |
| `sort LocalSet` | `SetId` (hand-rolled in `PROMELA-LOCAL-SET`; see note below) |
| `op _\|->_ : LVal Ref -> Env [ctor]` | Prelude MAP's `_\|->_` for `Map{LVal, Ref}` |
| `op _\|->_ : Int Value -> Store [ctor]` | Prelude MAP's `_\|->_` for `Map{Nat, Value}` |
| `op _\|->_ : Int Queue -> Network [ctor]` | Prelude MAP's `_\|->_` for `Map{Nat, Queue}` |
| `op _\|->_ : Id GotoCtx -> GotoMap [ctor]` | Prelude MAP's `_\|->_` for `Map{Id, GotoCtx}` |
| `op __ : Env Env -> Env [ctor assoc comm id: .Env]` | Prelude MAP's `_,_` for `Map{LVal, Ref}` |
| `op __ : Store Store -> Store [ctor assoc comm id: .Store]` | Prelude MAP's `_,_` for `Map{Nat, Value}` |
| `op __ : Network Network -> Network [ctor assoc comm id: .Network]` | Prelude MAP's `_,_` for `Map{Nat, Queue}` |
| `op __ : GotoMap GotoMap -> GotoMap [ctor assoc comm id: .GotoMap]` | Prelude MAP's `_,_` for `Map{Id, GotoCtx}` |
| `sort Proc` | `ProcCell` (AC multiset of `ProcCellItem` sub-cells) |
| `sort Procs` | `ProcsBag` (AC multiset of `proc(ProcCell)` wrappers) |
| `op proc : Int Id K LocalSet Env -> Proc [ctor]` | `op proc : ProcCell -> ProcsBagItem [ctor]`; sub-cells `pid : Int`, `name : Id`, `k : K`, `local : SetId`, `env : Map{LVal, Ref}` declared individually as `ProcCellItem` constructors |
| `op __ : Procs Procs -> Procs [ctor assoc comm id: .Procs]` | `op __ : ProcsBag ProcsBag -> ProcsBag [ctor assoc comm id: .ProcsBag]` |
| `sort Proctype` | `ProctypeCell` (AC multiset of `ProctypeCellItem` sub-cells) |
| `sort Proctypes` | `ProctypesBag` (AC multiset of `proctype(ProctypeCell)` wrappers) |
| `op proctype : Id Decls K GotoMap -> Proctype [ctor]` | `op proctype : ProctypeCell -> ProctypesBagItem [ctor]`; sub-cells `procName : Id`, `procParams : Decls`, `procBody : K`, `gotoMap : Map{Id, GotoCtx}` |
| `op __ : Proctypes Proctypes -> Proctypes [ctor assoc comm id: .Proctypes]` | `op __ : ProctypesBag ProctypesBag -> ProctypesBag [ctor assoc comm id: .ProctypesBag]` |
| `op intVal : Int -> Value [ctor]` | `op #_ : Int -> Value [ctor]` (per imp reference) |
| `op boolVal : Bool -> Value [ctor]` | `op #_ : Bool -> Value [ctor]` |
| `subsort Chan < Value` | `op #_ : Chan -> Value [ctor]` (Chan no longer auto-coerces to Value) |
| `sort KItem` | Removed; reference uses only `K` and `KLabel`. Effect markers moved to `KLabel` |
| `sort KProper` | Removed (was unused after KItem removal) |
| `sort KResult` | Removed |
| `op #rel : -> KItem [ctor]` | `op #rel : -> KLabel [ctor]` |
| `op #run : Id ValueList -> KItem [ctor]` | `op #run : Id ValueList -> KLabel [ctor]` |
| `op #wait : -> KItem [ctor]` | `op #wait : -> KLabel [ctor]` |
| `op #done : -> KItem [ctor]` | `op #done : -> KLabel [ctor]` |
| `op #init : Decls ValueList -> KItem [ctor]` | `op #init : Decls ValueList -> KLabel [ctor]` |
| `op #assign : Id Expr Expr -> KItem [ctor]` | `op #assign : Id Expr Expr -> KLabel [ctor]` |
| `op #send : Expr Exprs -> KItem [ctor]` | `op #send : Expr Exprs -> KLabel [ctor]` |
| `op #sendMsg : Chan Msg -> KItem [ctor]` | `op #sendMsg : Chan Msg -> KLabel [ctor]` |
| `op #recv : Expr Exprs -> KItem [ctor]` | `op #recv : Expr Exprs -> KLabel [ctor]` |
| `op #recvMsg : Exprs Msg -> KItem [ctor]` | `op #recvMsg : Exprs Msg -> KLabel [ctor]` |
| `op #freeze : -> KItem [ctor]` | `op #freeze : -> KLabel [ctor]` |
| `op #toss : Int -> KItem [ctor]` | `op #toss : Int -> KLabel [ctor]` |
| `subsort BranchSet < KItem` | `subsort BranchSet < K` (direct, since KItem dropped) |
| `subsorts Stmt Seq < KProper`, `subsorts KProper KResult < KItem`, `subsort KItem < K` | `subsorts Stmt Seq < KLabel` (single subsort step into KLabel; `KLabel < K` provided by `PROMELA-K`) |
| `op .ValueList : -> ValueList [ctor]` | `op nil : -> EmptyList [ctor]` (shared identity via subsort + `ditto`) |
| `op _,_ : ValueList ValueList -> ValueList [ctor assoc id: .ValueList prec 50]` | `op _,_ : ValueList ValueList -> ValueList [ctor ditto]` (attributes inherited from EmptyList's `_,_`) |
| `op .MsgList : -> MsgList [ctor]` | `nil` (shared with EmptyList) |
| `op _,_ : MsgList MsgList -> MsgList [ctor assoc id: .MsgList prec 50]` | `op _,_ : MsgList MsgList -> MsgList [ctor ditto]` |
| `op .Set : -> Set{Id} [ctor]`, `op __ : Set{Id} Set{Id} -> Set{Id} [ctor assoc comm id: .Set]`, `op _in_` | `op .SetId : -> SetId [ctor]`, `op __ : SetId SetId -> SetId [ctor assoc comm id: .SetId]` (no `_in_` since unused) |
| State accessors (`proctypes`, `procs`, `store`, `network`, `lock`, `active`, `nextLoc`, `nextPid`, `nextBCid`, `nextHCid`) — already removed in pre-Task-3 cleanup | Not present in new code; rules pattern-match cells directly |
| State mutators (`setProctypes`, `setProcs`, ..., `setNextHCid`) | Not present in new code |
| List-traversal helpers (`updateProc`, `updateProcEnv`, `updateProcLocal`, `getProc`, `getProctype`) — already removed pre-Task-3 | Not present |

## Preserved unchanged (Step 5)

These PROMELA-specific structures kept the same shape:

- `op @[_] : K -> BranchItem [ctor]` — load-and-fire branch wrapper
- `op .BranchSet : -> BranchSet [ctor]`, `op __ : BranchSet BranchSet -> BranchSet [ctor assoc comm id: .BranchSet]`
- `op gotoCtx : Bool K -> GotoCtx [ctor]`
- All effect-marker NAMES (`#rel`, `#run`, etc.) — only their host sort moved from `KItem` to `KLabel`
- `op ch : Int -> Chan [ctor]`, `op hs : Int -> Chan [ctor]` — Chan stays as a normal sort

## Function signature migrations

Functions in `promela-functions.maude` had their parameter types updated to match the new sort hierarchy. Logic preserved.

| Function | Old signature | New signature |
|---|---|---|
| `eval` | `Expr Env Store Network -> Value` | `Expr Map{LVal, Ref} Map{Nat, Value} Map{Nat, Queue} -> Value` |
| `evalList` | `Exprs Env Store Network -> ValueList` | `Exprs Map{LVal, Ref} Map{Nat, Value} Map{Nat, Queue} -> ValueList` |
| `isExec` | `BasicStmt Env Store Network Int -> Bool` | `BasicStmt Map{LVal, Ref} Map{Nat, Value} Map{Nat, Queue} Int -> Bool` |
| `gEnv` | `Env Set{Id} -> Env` | `Map{LVal, Ref} SetId -> Map{LVal, Ref}` |
| `effect` | unchanged signature; body uses `# 0` instead of `intVal(0)` |
| `purify` | unchanged signature; body uses `# PID` instead of `intVal(PID)` |
| `load` | unchanged signature; uses `# true` instead of `boolVal(true)` |
| `_==HChan_` | unchanged signature; pattern uses `# hs(I) ==HChan # hs(I)` |
| `intOf`, `boolOf` | retained; equations updated to match `# N` / `# B` patterns |

## Compromises and divergences

1. **Set{Id} hand-rolled rather than prelude SET**. Documented inline in `PROMELA-LOCAL-SET` (promela-state.maude). Maude's parametric SET imports a `_,_` operator whose `assoc comm` attributes conflict with our PROMELA-MSG `_,_` (which must remain non-commutative for ordered message contents). The conflict is not resolved by rename targets — Maude cross-compares rename targets against PROMELA-MSG's `_,_`. The IMP reference avoided this by never importing SET (IMP has no per-process variable-set cell). For us, hand-rolling `SetId` with `__` juxtaposition is the practical resolution.

2. **MAP/SET instantiations centralized**. All `MAP{X, Y}` and `LIST{X}` instantiations live in a single `PROMELA-MAPS` module rather than being scattered through PROC-CELL, PROCTYPE-CELL, and CONFIG. Reason: Maude's rename machinery does not cleanly separate the same renamed operator (`_;_`) when introduced by two distinct intermediate modules and re-imported downstream. Centralization sidesteps the issue.

3. **`isDoLoop` argument sort**. Originally `KItem -> Bool`. KItem was dropped, so the helper now takes `K`. The body adds a second equation to recognize `do _ od ~> K'` (continuation-prefix form) since K-as-a-sequence may not bottom out to a bare do-loop. This is a minor restructuring; logic preserved.

4. **`_,_` for ValueList/MsgList/Exprs unified via `ditto`**. Following the imp `EXP-LIST` pattern: a single `EmptyList` base sort declares `_,_` with attributes `[ctor assoc id: nil prec 70]`; ValueList, MsgList, Exprs all subsort from EmptyList and reuse `_,_` via `[ctor ditto]`. Identities unified to `nil`. This avoids the previous problem of three differently-identified `_,_` operators in the same kind.

5. **Rule pattern parenthesization**. Multiple rules in `promela-semantics.maude` use extra parens around map arguments (e.g., `env(((lval(X,I) |-> loc(L)), ENV))` instead of `env((lval(X,I) |-> loc(L)), ENV)`). Reason: `env : Map{LVal, Ref} -> ProcCellItem` is unary, but `,` is both the map separator and Maude's argument separator. Without extra parens, Maude tries to read `env(_, _)` as a binary application. The extra parens disambiguate.

## Matching Conventions (Task 4 Step A0)

Standard patterns used in every rule of `promela-semantics.maude`. Established before any rule was rewritten, then applied uniformly.

### Pattern 1 — Single-process rule

A rule that touches one specific process matches it inside the `procs` multiset, with a `PROC-REST : ProcsBag` remainder absorbing the other processes:

```maude
crl [rule-name] :
    procs(proc(pid(PID) name(NAME) k(K-PAT) local(LOCAL) env(ENV)) PROC-REST)
    CELL-1(...) CELL-2(...) CFG-REST
  =>
    procs(proc(pid(PID) name(NAME) k(K-PAT') local(LOCAL) env(ENV)) PROC-REST)
    CELL-1(...') CELL-2(...) CFG-REST
  if CONDITION .
```

Cells NOT modified by the rule are absorbed by `CFG-REST : KConfig`. All five sub-cells of `proc(...)` are listed explicitly because each process always carries exactly five sub-cells; the `ProcCell` AC operator matches by absorption against `.ProcCell` identity (which is empty in well-formed configurations).

### Pattern 2 — Two-process rule

`[hs-fire]` matches two distinct processes simultaneously. Maude's AC matching on the outer `procs(...)` multiset handles this by listing two `proc(...)` wrappers; AC matching binds the two PIDs to different elements:

```maude
crl [hs-fire] :
    procs(proc(pid(PID1) ...) proc(pid(PID2) ...) PROC-REST)
    ...
```

The Maude AC matcher guarantees the two `proc(...)` elements are distinct (otherwise both would unify into one element). No auxiliary structure required. Verified by Test 5 (send+recv across two procs).

### Pattern 3 — Nested sub-cell access

Within a `ProcCell` the sub-cells (`pid`, `name`, `k`, `local`, `env`) are AC-juxtaposed via `__` over `ProcCell` with identity `.ProcCell`. Rules that need to inspect/modify specific sub-cells list them explicitly:

```maude
proc(pid(PID) name(NAME) k(K) local(LOCAL) env(ENV))
```

The listing must enumerate all five sub-cells because well-formed `ProcCell`s always contain exactly five. There is no `PROC-CELL-REST : ProcCell` remainder pattern currently needed since no rule wants "any process with at least these sub-cells" — every rule reads/writes the full set.

### Pattern 4 — `crl` vs `rl` choice

Re-evaluated each rule's classification under the new structure. Outcome: **no classification changes**. Rationale per family:

- **`[fire]`, `[hs-fire]`, `[toss1]`, `[break-prop]`, `[assign-eval-index]`, `[assign-eval-value]`, `[assign-store]`, `[assign-chan]`, `[send-eval-chan]`, `[recv-eval-chan]`, `[run-eval]`** — all `crl` because each has a real condition (`isExec(...)`, `intOf(VIDX) == I`, `not(E :: Value)`, `not(VR :: Value)`, etc.). No change.
- **`[rel]`, `[loose]`, `[toss2]`, `[send-push]`, `[recv-pop]`, `[recvMsg-*]`, `[goto]`, `[break]`, `[run]`, `[done]`, `[init-done]`, `[freeze]`, `[load-*]`** — all `rl` because they are structural transformations with no conditional guards. No change.
- **`[loose]`** — investigated whether multiset matching admits a cleaner expression than the old "rule with no `[owise]`". Conclusion: no — the K `[owise]` was "fire only when no other rule matches the same K-head", which requires negative pattern matching across the whole configuration. Maude does not support that. The `rl` stays without `[owise]`, retaining the documented nondeterministic-race compromise.
- **`[load-*]` family** — investigated converting to `eq`. Conclusion: keep as `rl`. K source declares the corresponding `[spload-*]` rules as actual rules (not `[function]`), so the load-* family stays as `rl`. The equational `load` function reduces inside the rule's RHS.
- **Assignment family** — multi-step decomposition preserved as four `crl`s; no multiset-matching shortcut found that would collapse them into one rule while keeping the K semantics.

### Pattern 5 — PID-based controller lookup

`[run]` and `[goto]` look up a proctype by name (a Maude variable that occurs both in the matched `proctype(procName(X) ...)` cell and in the matched `proc(name(X) ...)` cell — wait, no, `[run]` matches proctype by `procName(X)` where `X` is the spawn target, and the calling proc has its own `name(NAME)`). The two `X` occurrences in `[run]` (one in `proctype(procName(X) ...)`, one in `#run(X, VL)`) bind together via non-linear matching — Maude handles this natively.

For controller-process patterns (`pid = -1` in K source), no rule currently special-cases pid -1 in our port (controller-only semantics were not ported). If/when ported, the pattern would be `proc(pid(-1) ...)` which AC-matches exactly one process with that PID.

## Verification

```
$ maude -no-banner k-prelude.maude       → 0 errors, 0 warnings
$ maude -no-banner promela-state.maude   → 0 errors, 0 warnings
$ maude -no-banner promela-functions.maude → 0 errors, 3 warnings (pre-existing benign `_-_` kind-level ambiguities at lines 67, 148, 231)
$ maude -no-banner promela-semantics.maude → 0 errors, 3 warnings (same 3 inherited)
```

Test suites (`tests.maude`, `tests-extended.maude`) are NOT expected to pass after this task. Task 4 will repair them along with the rule logic.

## ESCALATE items

None. All rules were rewritten mechanically against the new multiset pattern. No rule required a Maude language feature that was unavailable.

---

# Task 4 Summary — Rule Rewrite and Test Recovery

## Part A — Rule rewriting

### A0 Matching conventions
Documented above under "Matching Conventions (Task 4 Step A0)" with five patterns: single-process, two-process, nested sub-cell, `crl`/`rl` choice, PID-based controller lookup.

### A1 LHS/RHS multiset form
All 24 rules in `promela-semantics.maude` rewritten to the multiset configuration. Cells not modified by a rule are absorbed by `CFG-REST : KConfig`; the `procs(...)` and `proctypes(...)` wrappers use `PROC-REST : ProcsBag` / `PT-REST : ProctypesBag` remainders for AC matching.

### A2 Classification re-evaluation
No `eq` ↔ `rl`/`crl` reclassifications. Documented in "Matching Conventions" Pattern 4: every rule's old classification is the right classification under the new structure too. `[loose]` and `[load-*]` were specifically re-examined and confirmed unchanged.

### A3 Effect markers
All effect markers (`#assign`, `#send`, `#sendMsg`, `#recv`, `#recvMsg`, `#run`, `#wait`, `#done`, `#init`, `#freeze`, `#toss`, `#rel`) verified to work under the new structure. Their host sort migrated from `KItem` to `KLabel`, but their semantic role is unchanged.

### A4 Preserved design decisions
All preserved:
- `[loose]` does not carry `[owise]` (Maude rules cannot)
- `subsort Expr < BasicStmt` for guard expressions
- `#send`/`#recv` first argument is `Expr` (covers `Vref`, `Value`, `Chan`-via-`#`)
- `VIDX`/`VOLD` naming in `[assign-store]` (avoids non-linear matching)
- `VR : Vref` widening in `[send-eval-chan]`/`[recv-eval-chan]`

### Late-discovered fix: SetId separator
The hand-rolled `SetId` initially used `__` (juxtaposition), which collided with `BranchSet`'s `__` (different identity, same kind). The collision prevented Maude's AC normalization from absorbing `.BranchSet`, causing `[fire]` to mis-match BranchSet-wrapped K continuations. Fix: switched SetId to use `_,_` via ditto from the shared `EmptyList`/`nil` pattern (PROMELA-MSG). This costs strict commutative-set semantics (acceptable for our use of SetId as a name collection for `gEnv` filtering) and resolves the AC normalization bug.

Side effect of the fix: the three pre-existing benign `_-_` kind-level ambiguity warnings in `promela-functions.maude` (lines 67, 148, 231) **disappeared**. The new EmptyList-ditto pattern apparently disambiguates the kind for `_-_` enough that Maude no longer reports the alternate kind parse. Backend now loads with **zero warnings and zero errors** across all four files.

## Part B — Test recovery

### B1 tests.maude
Migrated all 12 test groups (28 result lines total). Surface-syntax updates:
- `intVal(N)` → `# N`
- `boolVal(B)` → `# B`
- `.Env`, `.Store`, `.Network` → `empty` (prelude MAP identity)
- `.Set` → `nil` (shared EmptyList identity via SetId)
- `.MsgList`, `.ValueList` → `nil`
- `.Procs` → `.ProcsBag`, `.Proctypes` → `.ProctypesBag`
- `state(...)` flat record → multiset of cells (`procs(...) proctypes(...) store(...) ...`)
- `proc(P, N, K, L, E)` → `proc(pid(P) name(N) k(K) local(L) env(E))`
- `proctype(N, D, B, G)` → `proctype(procName(N) procParams(D) procBody(B) gotoMap(G))`

All 12 tests produce semantically-correct results. Test 11 was simplified from `@[ # true ~> .K ] .BranchSet) ~> .K` to bare `@[ # true ]` (equivalent under AC normalization with `.BranchSet`/`.K` identities now working correctly).

### B2 tests-extended.maude
Migrated all 10 test groups (13 result lines total: 9 `rew` + 4 `matchMsg red` + 1 `search`). Same surface-syntax updates as B1.

Test E4's expectation comment was updated: the test bypasses `[fire]`/`isExec` (calls `#send` directly in K continuation), so capacity isn't enforced by `[send-push]` alone — capacity check lives in `isExec`, consulted only by `[fire]`. Updated comment to reflect actual behavior.

Test E7 (search) target pattern was rewritten as a multiset of cells (instead of the old `state(...)` constructor). Search finds Solution 1 at state 44 with L:Lock = #none.

### B3 Baseline captured

```
tests.maude:         28 result lines, all semantically correct
tests-extended.maude: 13 result lines (1 search Solution + 12 reduce/rewrite),
                     all semantically correct
```

Semantic verification (extended tests):
- E1 multi-step assign: store(1 |-> # 6) ✓
- E2 atomic: store((1 |-> # 1, 2 |-> # 1)) ✓
- E3 if-selection: store(2 |-> # 2) ✓ (second branch fired)
- E4 channel sends: 3 messages pushed (direct-#send path bypasses cap; see updated comment)
- E5 matchMsg: true / false / true / false ✓
- E6 two-proc interleave: store((1 |-> # 1, 2 |-> # 1)) ✓
- E7 search: Solution 1 found at state 44 ✓
- E8 mutex: P0's cs updated, P1 blocked at @[...] ✓
- E9 goto: store(1 |-> # 42) ✓
- E10 do-loop+break: k(.K) ✓ (full reduction, no .BranchSet residue)

## Part C — Final verification

| Check | Result |
|---|---|
| `k-prelude.maude` loads | 0 errors, 0 warnings |
| `promela-state.maude` loads | 0 errors, 0 warnings |
| `promela-functions.maude` loads | 0 errors, 0 warnings |
| `promela-semantics.maude` loads | 0 errors, 0 warnings |
| `tests.maude` | 28 result lines, all correct |
| `tests-extended.maude` | 13 result lines, all correct |

## Known open issues — status after restructuring

1. **`[loose]` losing `[owise]` semantics** (resolved later, see "2026-10-14" at the end): NOT resolved by restructuring. Maude rules still cannot carry `[owise]`. `[loose]` retains its documented non-deterministic-race compromise. The multiset matching did not provide a cleaner expression.
2. **Latent `[hs-fire]` `_[_]` overload bug** (Task 1 audit Part D, item D8): UNCHANGED. `[hs-fire]` still uses `X[I] ! EL1` (Int-index pattern). Fix would require widening to `VR : Vref` analogous to `[send-eval-chan]`. Not done in Task 4 per the constraint to preserve known design decisions.
3. **Ambiguous send/recv parsing** (Task 1 audit Part D, item D37 - `[load-basic]` coverage): UNCHANGED. The `subsort Expr < BasicStmt` chain still allows `[load-basic]` to fire on bare Values. No test exercises this boundary; behavior unverified.
4. **Benign `_-_` kind-level warnings** in PROMELA-EVAL/PURIFY/EFFECT: **resolved as a side effect of the EmptyList-ditto unification**. The new sort structure for ValueList/Exprs/MsgList apparently disambiguates the kind for `_-_` and Maude no longer reports the alternate parse. Backend is now warning-free.

## Backend ready for frontend implementation
After Task 4, the backend exhibits the multiset configuration pattern faithfully, all rules match against the nested cell hierarchy, tests pass, and load is clean. Frontend implementation can resume.

## 2026-10-14 — `[loose]` carries the owise condition; loading keeps the atomic context

Two departures from `promela.k` are removed. Both made the lock leave its holder in the middle
of a statement or of an atomic block, so atomic blocks were not atomic (bakery's mutual
exclusion, `spin-atomic1`, `spin-atomic2` and `paper-spin-atomic-handshake` had spurious
violations).

- **`[loose]`** (promela.k:376, `[owise]`). K tries an owise rule only when no other rule
  applies to the configuration. A Maude rule cannot carry `owise`, but a condition can state
  it: `crl [loose] : ... if not canStep([lock(@ PID) CFG-REST])`. `canStep(GS)` is true when a
  rule other than `[loose]` can rewrite `GS`. It has one equation per such rule, with the
  left-hand side and the condition of the rule, and `canStep(GS) = false [owise]`; an equation
  can carry `owise`. The equations are generated from the rules by
  `scripts/gen_promela_act.py` into the block at the end of `PROMELA-SEMANTICS`, and
  `scripts/check_promela_act.py` compares `canStep` with the rules on every configuration it
  explores. `META-LEVEL` is not imported. This supersedes "Pattern 4" and "A4" above, which
  recorded that `[loose]` stays an `rl` without the condition.
- **`[load-*]`** (promela.k:680-691 `[spload-true]`, 719-726 `[spload-atomic]`). K loads a
  statement with `spload(true, ...)` when the loading process holds the lock and with
  `spload(false, ...)` when the lock is free. The seven `[load-*]` rules passed `false`
  always, which put a `#rel` after every statement of an atomic block from the second one on.
  They now pass `LCK == @ PID`. The scheduling around `spload` (`#normalize`, the controller
  process, the priorities) is still not ported.
- `[rel]` (promela.k:373) is unchanged: it frees the lock whoever holds it, as in K.

Effect on the state spaces: a statement is no longer interleaved between `[fire]` and its
effect, so every benchmark has fewer states (dining-philosophers-2 1473 -> 185, filter-2procs
801 -> 336, qlock-2procs 782 -> 255, onethirdrule-2procs 749 -> 299, chan-buffered-1 166 -> 58,
bakery-2procs 6214 -> 268). Verdicts are unchanged except where they depended on atomicity.
New tests: `tests-atomic.maude` (13 searches).
