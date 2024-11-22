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

### Lock
- #acquire
- #release

- atomic { SL }
>> assume atomic is not nested. (this can be ensured by deleting inner atomics without changing the behaviour)

e.g
```
k(atomic { SL } ~> K) => k(#acquire ~> SL ~> #release ~> K)
```

### Selection
- #release-after
```
k(#release-after ~> S ~> K) => k(S ~> #release ~> K)
```


### Notes
* why flattening may not work
>> consider if ... :: do ... od :: ... fi. In this case, the structure of do ... od must be preserved.
otherwise, when do branch is taken and reached the end, we don't know where to go back.
