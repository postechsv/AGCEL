# SKELETON_NOTES.md

Architectural rationale for `semantics/promela-skeleton.maude`. Each design choice is justified by reference to the conventions documented in `imp/NOTES.md` and to the K source at `semantics/promela.k`.

The skeleton is structure-only. It declares the configuration hierarchy, parameterized-map instantiations, the Value-wrapper convention, and one trivial rule (so the system module qualifies as `mod`). It declares no effect-marker semantics, no functions, and no substantive rules. The skeleton lives alongside the existing four-file backend without replacing it; the existing tests continue to pass.

---

## Configuration sort

**Choice**: declare `sort KConfig` as the AC-multiset supersort, with identity `none` and operator `__`. Cells are `KConfigItem`, a subsort of `KConfig`.

**Source**: `imp/NOTES.md §1`, citing `imp/k-prelude.maude` lines 11–20. The reference's `K-CONFIG` module uses exactly this shape:

```
sorts KConfig .
op none : -> KConfig [ctor] .
op __ : KConfig KConfig -> KConfig [ctor assoc comm id: none] .
sort KConfigItem .
subsorts KConfigItem < KConfig .
```

**No `State` sort.** The current backend's `sort State` with a 10-arg `state(...)` constructor is the structural error this restructuring exists to address. The skeleton's `KConfig` is the configuration; nothing wraps it.

---

## Cell payload sorts

**Choice**: each top-level cell holds the parameterized prelude map / list directly (e.g., `store` holds `Map{Nat, PSKEL-Value}`). No intermediate sort like `Env`, `Store`, `Network`, `GotoMap` is introduced.

**Source**: `imp/NOTES.md §2`, citing `imp/problem2-sol.maude` lines 47–54, where `env`, `store`, `genv` each hold `Map{...}` directly with no wrapping sort.

**Mapping from K to Maude prelude module**:
- `<store>` (K's Map) → `Map{Nat, PSKEL-Value}`
- `<network>` (K's Map) → `Map{Nat, PSKEL-Queue}`
- `<env>` (K's Map) → `Map{PSKEL-LVal, PSKEL-Ref}` (per process)
- `<gotoMap>` (K's Map) → `Map{PSKEL-Id, PSKEL-GotoCtx}` (per proctype)
- `<local>` (K's Set) → `Set{PSKEL-Id}` (per process)
- `<active>` (K's List) → `List{Int}` (top-level)

Each parameterized instantiation uses `* (op _,_ to _;_)` to rename the default `_,_` map/set separator, matching `imp/problem2-sol.maude` line 24. This avoids collision with the value-list `_,_` we will add for messages in Task 3.

---

## Value wrapper convention

**Choice**: a single overloaded prefix operator `#_` with three arity-1 overloads (`Int -> Value`, `Bool -> Value`, `Chan -> Value`).

**Source**: `imp/NOTES.md §4`, citing `imp/imp-syntax.maude` lines 1–9:

```
fmod IMP-VALUE is
  protecting INT .
  sorts Value .
  op #_ : Bool -> Value [ctor] .
  op #_ : Int -> Value [ctor] .
endfm
```

The reference comment explicitly documents the reason: *"declaring Bool and Int as subsorts of Value is not allowed in Maude"*. We follow the convention verbatim and add a third overload for `Chan` (which IMP does not need but PROMELA does).

This replaces the current backend's two differently-named wrappers (`intVal`, `boolVal`) with a single operator name. Parsing of `# 5`, `# true`, `# ch(0)` is verified to produce `Value` in the load test.

---

## Process / proctype grouping

**The ambiguous case the task spec acknowledges.** The K source has `<proc multiplicity="*">`, meaning each process is a separate cell containing five sub-cells. In Maude, two designs are possible:

### Option A — chosen: wrapped per-process bag

```
op procs : ProcsBag -> KConfigItem [ctor] .
op proc : ProcCell -> ProcsBagItem [ctor] .  -- one wrapper per process
sort ProcCell .                              -- inner multiset of 5 sub-cells
op .ProcCell : -> ProcCell [ctor] .
op __ : ProcCell ProcCell -> ProcCell [ctor assoc comm id: .ProcCell] .
op pid : Int -> ProcCellItem [ctor] .
op name : Id -> ProcCellItem [ctor] .
... etc.
```

A concrete state with two processes parses as:
```
procs(proc(pid(0) name('Main) k(.K) local(empty) env(empty))
      proc(pid(1) name('W)    k(.K) local(empty) env(empty)))
```
The outer `__` (over `ProcsBag`) keeps the two `proc(...)` wrappers separate. Inside each wrapper, the inner `__` (over `ProcCell`) holds the 5 sub-cells.

### Option B — rejected: flat sub-cells inside `procs`

The task spec also describes a reading where `ProcCell` is itself the multiset sort and "multiple processes coexist as a flat multiset inside `procs`." Taken literally, this would have:

```
op procs : ProcCell -> KConfigItem [ctor] .  -- no per-process wrapper
op pid : Int -> ProcCellItem [ctor] .         -- pid lives directly in ProcCell
...
```

A two-process state would then be:
```
procs(pid(0) name('Main) k(.K) local(empty) env(empty)
      pid(1) name('W)    k(.K) local(empty) env(empty))
```

**Why Option B is rejected**: with the AC `__` over `ProcCell`, the boundary between processes dissolves. Maude's matcher cannot tell which `pid` belongs to which `name` / `k` / `local` / `env`. A rule like `[fire]` that needs to pattern-match one process's pid against its own k cell would have no way to bind them as a unit. The K source's `multiplicity="*"` discipline is exactly the per-process grouping that Option A preserves.

**Why Option A does not violate the "no named record sort" prohibition**: `proc` is an operator (a cell constructor in the K sense), not a sort. The sort is `ProcCell` (the inner multiset). Multiset structure is preserved; `proc` plays the role K plays with `<proc>...</proc>` brackets — a delimiter, not a record.

### Gap in the `imp/` reference

`imp/NOTES.md` flags this explicitly: the IMP / IMP# reference does not show a direct example of nested sub-configuration with `multiplicity="*"`. The closest analog is `FuncDecl` in `imp#-syntax.maude` line 70:
```
op __ : FuncDecl FuncDecl -> FuncDecl [ctor assoc comm id: empty] .
```
where each function is `function F(XL) {CODE}` — an *atomic* term packaged as a single constructor. The reference has no example of grouping a multiset-of-sub-cells under a wrapper because IMP's functions have a fixed-arity body (one `Block`), not a variable-arity bag.

In the absence of a direct reference, we apply the general principle of `imp/NOTES.md §1` (a configuration is a multiset of `KConfigItem` constructors) recursively. The `proc(...)` and `proctype(...)` wrappers extend the same pattern one level inward: the inner sub-configuration is itself a multiset, wrapped in a constructor that the outer multiset treats as a single element. This is the standard K cell discipline.

The same wrapper pattern is applied to proctypes:
```
op proctypes : ProctypesBag -> KConfigItem [ctor] .
op proctype : ProctypeCell -> ProctypesBagItem [ctor] .
sort ProctypeCell .
... etc, with sub-cells procName / procParams / procBody / gotoMap.
```

---

## Effect markers

**Choice**: declared two illustrative markers (`#rel`, `#wait`) as `KLabel` constructors. No semantics; only present to demonstrate where Task 3 will populate the full set.

**Source**: `imp/NOTES.md §5`, citing the reference convention `subsort Exp Block < KLabel` (`imp/imp-basic-semantics.maude` line 29). Per the task spec §3, this skeleton declares no functions or substantive rules; effect markers are stubbed only.

The current backend declares effect markers (`#rel`, `#run`, `#wait`, `#done`, `#init`, `#assign`, `#send`, `#sendMsg`, `#recv`, `#recvMsg`, `#freeze`, `#toss`) all as `KItem`. The reference convention is `KLabel`; the skeleton adopts `KLabel` to match.

---

## Syntax sort lifting (`KLabel`, not `KItem`)

**Choice**: declare `sort KLabel` as a subsort of `K`, following `imp/k-prelude.maude` lines 6–7. Do not declare `sort KItem`.

**Source**: `imp/NOTES.md §5`. The reference's `K-COMPUTATION` module has exactly:
```
sort K .
op .K : -> K [ctor] .
op _~>_ : K K -> K [ctor prec 70 assoc id: .K] .
sorts KLabel .
subsorts KLabel < K .
```

No intermediate `KItem`. Source-language sorts (and effect markers) subsort directly into `KLabel`.

The current backend's `sort KItem` with `KItem < K` and `KProper / KResult < KItem` does not appear in this reference. The skeleton drops `KItem` and the `KProper`/`KResult` two-level discipline. Task 3 may revisit if a concrete need surfaces, but the reference's IMP semantics works without them.

---

## `view` declarations

**Choice**: declared seven views for the user sorts that appear as map keys / values: `PSKEL-Id`, `PSKEL-LVal`, `PSKEL-Ref`, `PSKEL-Value`, `PSKEL-Queue`, `PSKEL-GotoCtx`. The view names are prefixed `PSKEL-` to avoid colliding with Maude prelude views (e.g., `Nat`, `Int`, `Qid`, `Bool` already exist in the prelude).

**Source**: `imp/NOTES.md §3`, citing `imp/imp-syntax.maude` lines 20–27. Each view binds the parameter sort `Elt` (from `TRIV`) to the user sort.

`Nat` and `Int` come prepackaged from the Maude prelude, so we use those directly without re-declaring views.

---

## Module decomposition

The skeleton is split into 11 modules:

| Module | Role | LoC range |
|---|---|---|
| `PSKEL-VALUE` | Value sort + Chan + `#_` wrappers | ~12 |
| `PSKEL-NAMES` | Id, LVal | ~12 |
| `PSKEL-CHAN-REF` | Loc, Ref | ~10 |
| `PSKEL-LOCK` | Lock | ~7 |
| `PSKEL-QUEUE` | Msg, MsgList, Queue | ~10 |
| `PSKEL-K` | K, KLabel, Decls, GotoCtx | ~18 |
| (7 views) | TRIV-to-user-sort views | ~21 |
| `PSKEL-K-CONFIG` | KConfig, KConfigItem | ~11 |
| `PSKEL-PROC` | ProcCell + per-process sub-cells | ~22 |
| `PSKEL-PROCTYPE` | ProctypeCell + per-proctype sub-cells | ~19 |
| `PSKEL-PROMELA-CONFIG` | Top-level cells + procs/proctypes bags | ~42 |
| `PSKEL-EFFECT-MARKERS` | `#rel`, `#wait` (stubs) | ~10 |
| `PSKEL-SYSTEM` | System module wrapping all the above + 1 trivial rule | ~13 |

Each module imports exactly what it needs. The `protecting` chains form a DAG with `PSKEL-SYSTEM` at the root. This mirrors the reference's modular split in `problem2-sol.maude`, where `CALL-STACK`, `IMP#-CONFIG`, `IMP#-RULES` are layered.

---

## Trivial sanity rule

**Choice**: declared one `rl [skel-sanity]` that performs an idempotent rewrite on the `active` cell. The rule never fires productively; it exists solely so that `PSKEL-SYSTEM` qualifies as `mod` and Maude's parser exercises the rule-position path.

**Source**: task spec §3 explicitly allows "at most one trivial rule (e.g., a constant transition) to verify the file actually loads as a `mod` and not just an `fmod`." Task 3 will populate this module with the real rule set.

---

## Load verification

```
$ maude -no-banner promela-skeleton.maude
```

Produces:
- **Zero warnings**
- **Zero errors**
- ~12 advisories, all of the form *"sort X has been imported from both PSKEL-QUEUE and PSKEL-QUEUE"* and similar for `MsgList`, `Queue`, `Msg`, `mq`, `.MsgList`. These arise because `PSKEL-PROMELA-CONFIG` reaches `PSKEL-QUEUE` by two paths: directly (`protecting PSKEL-QUEUE`) and transitively through `MAP{Nat, PSKEL-Queue}` (whose view target also pulls in `PSKEL-QUEUE`). Maude reports the redundancy but resolves it correctly; the operators are not duplicated.

Same pattern as the load-bearing `__` advisories already accepted in the working backend (`promela-state.maude` line 11 etc., per the existing project record). Removable if needed by reshaping the import graph, but not blocking for skeleton verification.

### Smoke-parse test
The following parses successfully as `KConfig`:
```
proctypes(.ProctypesBag)
procs(proc(pid(0) name('Main) k(.K) local(empty) env(empty)))
store(empty) network(empty) lock(#none) active(nil)
nextLoc(0) nextPid(1) nextBCid(0) nextHCid(0)
```
Each of `# 5`, `# true`, `# ch(0)` parses as `Value`.

The existing backend test suites (`tests.maude`: 28 result lines; `tests-extended.maude`: 12 result lines) are unaffected.

---

## What is NOT in this skeleton (deferred to Tasks 3–4)

- The full set of effect markers (`#run`, `#done`, `#init`, `#assign`, `#send`, `#sendMsg`, `#recv`, `#recvMsg`, `#freeze`, `#toss`).
- Source-language syntax (Exp, Stmt, Decl, Block, BasicStmt, GotoStmt, Vref, Optn, Optns, etc.).
- Functions (`eval`, `evalList`, `purify`, `isExec`, `effect`, `split`, `load`, `==HChan`, `gEnv`).
- Rules (`[fire]`, `[hs-fire]`, `[rel]`, `[loose]`, `[toss1/2]`, assign family, send/recv family, recvMsg family, goto/break, run/done/init-done, freeze, load-*).
- The `MsgList` / `ValueList` separator unification carried over from the existing backend.

Tasks 3 and 4 will port these against the cell hierarchy this skeleton establishes.
