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

Hence, those side-effects for selection and atomic should be 'delayed' until a first basic actions is found,
then should take place 'simultaneously' with the main effect of that basic action.

- main effect : basic actions
- default side-effect : selection
- optional side-effect : no-effect(.E) / atomic(@)

K ::= Set(Option)

* lift
```
k(SL) => k(< . :: SL>)
```

### Basic Actions
```
k(< E :: X = V ; SL > OPTS) lock(L) env(ENV) store(STORE)
=>
k(< .E :: SL > .OPTS) lock(f(L, E)) env(ENV') store(STORE')
```

### Lock
- S ::= #acquire | #release

- atomic { SL } = #acquire ; SL ; #release
>> assume atomic is not nested. (this can be ensured by deleting inner atomics without changing the behaviour)

* acquire
Note lock() is not affected 'yet'.
```
k(< .E :: #acquire ; SL > OPTS)
=>
k(< @ :: SL > OPTS)
```

* release
`#release` can be executed on its own, because the lock is acquired.
```
pid(I) k(< .E :: #release ; SL >) lock(I)
=>
pid(I) k(< .E :: SL >) lock(none)
```


### Selection
* select
```
k((:: if OPTL fi) OPTS)
=>
k((:: if OPTL fi) OPTS)
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
