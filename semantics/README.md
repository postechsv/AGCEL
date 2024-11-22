# Semantics for Promela as our modeling language

## Syntax
- Statement(S) ::= assign | selection | loop
- Sequence(SL) ::= S | S ; SL

## Configuration
Lock(PID or none)
Process(
  k(K)
  env()
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
- #rel


### Notes
* why flattening may not work
>> consider if ... :: do ... od :: ... fi. In this case, the structure of do ... od must be preserved.
otherwise, when do branch is taken and reached the end, we don't know where to go back.
