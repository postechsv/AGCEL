# Semantics for Promela as our modeling language

## Syntax
- Action(A) ::= assign | chanop
- Statement(S) ::= A | selection | loop | atomic
- Sequence(SL) ::= S | S ; SL

## Configuration
Lock(PID or none)
Process(
  k(K)
  env(ENV)
)


## Semantics

### Chanllenges
- nested loop, selection, and atomic
- interference when rendezvous operations are at the guards of selections.
- executability
- state-space explosion

### Lock
- #acquire
- #release

- atomic { SL }
>> assume atomic is not nested. (this can be ensured by deleting inner atomics without changing the behaviour)

e.g
```
k(atomic { SL } ~> K) => k(#acquire ~> SL ~> #release ~> K)
```

acquire only if executable
```
pid(I) k(#acquire ~> SL ~> K) Lock(none) => pid(I) k(SL ~> K) Lock(I)
if executable(S)
```

release temporarily if blocked
```
pid(I) k(S ~> K) Lock(I) => pid(I) k(#acquire ~> S ~> K) Lock(none)
if not executable(S)
```

### Selection
- #release-after
Recall actions(A) is a simple atomic transition in the SPIN engine.
```
k(#release-after ~> A ~> K) => k(A ~> #release ~> K)
```

- if OPTIONS fi
Unlocked case)
Note that executable-branch always returns a sequence of form A ; SL.
This is why prepending #release-after makes sense.
```unlocked
pid(I) k(if OPTIONS fi ~> K) Lock(none) => pid(I) k(#release-after ~> executable-branch(OPTIONS) ~> K) Lock(I)
if executable(if OPTIONS fi)
```

Locked case)
```locked
pid(I) k(if OPTIONS fi ~> K) Lock(I) => pid(I) k(executable-branch(OPTIONS) ~> K) Lock(I)
if executable(if OPTIONS fi)
```


### Loop
- do OPTIONS od
```
k(do OPTIONS od ~> K) => k(if OPTIONS fi ~> ~> do OPTIONS fi ~> K)
```

## Notes
* why flattening may not work
>> consider if ... :: do ... od :: ... fi. In this case, the structure of do ... od must be preserved.
otherwise, when do branch is taken and reached the end, we don't know where to go back.
