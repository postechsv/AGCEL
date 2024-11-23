# Semantics for Promela as our modeling language

## Syntax
- Basic Actions corresponds to basic transitions in the SPIN engine.
Hence basic actions should be executed atomically.
>> A ::= assign | chanop
- Statements
>> S ::= A | selection | loop | atomic
- Sequences
>> SL ::= S | S ; SL

## Configuration
lock(PID | none)
store(STORE)
process(
  k(K)
  acq(BOOL)
  env(ENV)
)


## Semantics

### Executability

### Side-effects
The trickiest part of defining the semantics of Promela is in how to define the semantics of atomic and selection constructs.
Here are their behaviours:
- The behaviour of `atomic{ SL }` is: first acquire the global lock when SL is executable, then run SL, and finally release the lock.
- The behaviour of `if (:: SL_1) (:: SL_2) ... (:: SL_n) fi` is: first choose an executable option :: SL_i, then run SL_i.

Note that the 'acquire' and the 'choose' parts are not basic actions.
Rather, they are 'side-effects' that are attached to a 'main effect' of a basic action.
So if they should not be a valid transition on their own.

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
Semantics for Selection is very nontrivial.
The difficulty comes from the fact that in order to perform selection, we must:
1) test if there is an executable branch
2) if there is one, choose one
3) execute the very first action in the chosen branch
Since there should be no interleaving between step 1), 2) and 3) we must ensure atomicity.
If we try to define a single rule that does 1), 2) and 3) all at once,
we get stuck because we don't know how to execute the action in step 3).
Then, we are forced to define the selection step separately, while ensuring atomicity.
Hence we utilize the global lock.

Unlocked case)
Note that executable-branch always returns a sequence of form A ; SL.
This is why prepending #release-after makes sense.
Also, the condition executable(if OPTIONS fi) ensures that executable-branch(OPTIONS) is well-defined.
```unlocked
pid(I) k(if OPTIONS fi ~> K) Lock(none) => pid(I) k(#release-after ~> executable-branch(OPTIONS) ~> K) Lock(I)
if executable(if OPTIONS fi)
```

Locked case)
WARNING: even if the lock is acquired, we cannot ensure the executability. So executability check is necessary.
```locked
pid(I) k(if OPTIONS fi ~> K) Lock(I) => pid(I) k(executable-branch(OPTIONS) ~> K) Lock(I)
if executable(if OPTIONS fi)
```

Note that the difference is only in #release-after.
Actually, we can combine the above two cases for unlocked and locked, into a single rule.
- Define ifPrefix(L) := (L == none ? #release-after : .K)

Combined Rule)
```locked
pid(I) k(if OPTIONS fi ~> K) Lock(L) => pid(I) k(ifPrefix(L) ~> executable-branch(OPTIONS) ~> K) Lock(I)
if executable(if OPTIONS fi) and (L = none or L = I)
```


### Loop
- do OPTIONS od
```
k(do OPTIONS od ~> K) => k(if OPTIONS fi ~> ~> do OPTIONS fi ~> K)
```

### Goto

## Notes
* why flattening may not work
>> consider if ... :: do ... od :: ... fi. In this case, the structure of do ... od must be preserved.
otherwise, when do branch is taken and reached the end, we don't know where to go back.
