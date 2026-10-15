#!/usr/bin/env python3
"""Generate N-process PROMELA analysis files in the layout of benchmarks-promela/*-analysis.maude
(modules MODEL, MODEL-PLAIN, MODEL-FOCUS and a *-PROMELA-TEST module with init and goal).

Families (FAMILY: test module, goal):
  qlock         FIFO-queue mutex; goal someInCS
  dining        philosopher i takes fork i, then fork (i+1) mod N; goal deadlock (every process blocked)
  filter        filter lock with N-1 levels; goal someInCS
  onethirdrule  one round: send, wait for all N arrivals, adopt the minimum, decide; goal allDecided
  exfourc       Holzmann's exercise ex.4c, a faulty mutual exclusion algorithm from the literature;
                goal twoInCS (the assertion of the original fails)

For N = 2, qlock, dining and onethirdrule are the hand-written benchmarks; filter is the same
algorithm in level/victim form (Peterson's flag/turn form in filter-analysis.maude).
The vocabulary follows the hand-written files: one varEq per non-zero value of each shared
variable, the per-process predicates of the family, allDone and the goal; pidProps lists
assignP for every variable, sendP/recvP for every channel, and locked.

  gen_promela_bench.py <family> <n> <out.maude>
"""
import os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMILY = {'qlock': ('QLOCK-PROMELA-TEST', 'someInCS'), 'dining': ('DINING-PROMELA-TEST', 'deadlock'),
          'filter': ('FILTER-PROMELA-TEST', 'someInCS'), 'onethirdrule': ('OTR-PROMELA-TEST', 'allDecided'),
          'exfourc': ('EXFOURC-PROMELA-TEST', 'twoInCS')}
# further goals of a family, selected as <family>-<n>.<variant>: (test module, goal)
VARIANT = {('qlock', 'mutex'): ('QLOCK-PROMELA-MUTEX', 'twoInCS'), ('filter', 'mutex'): ('FILTER-PROMELA-MUTEX', 'twoInCS')}

HEAD = f'''load {REPO}/semantics/promela-agcel.maude .
load {REPO}/semantics/promela-obs-generated.maude .
load {REPO}/benchmarks-promela/focus-mt.maude .
'''

OBS = '''
mod MODEL-PLAIN is
  pr MODEL .
  var S : State .
  eq obs(S) = < obsUpto(S, plainProps) > .
endm

mod MODEL-FOCUS is
  pr MODEL .
  var S : State .
  eq obs(S) = < obsUpto(S, pidObs ; plainProps) > .
endm
'''


def procs(n):
    return '\n'.join(f"      proc(pid({i}) name('P{i}) k( body{i} ~> .K ) local(nil) env(env{i}))" for i in range(n))


def envs(n):
    return '\n'.join(f'  op env{i} : -> Map{{PROMELA-LVal, PROMELA-Ref}} .\n  eq env{i} = env .' for i in range(n))


def ids(n):
    return ' '.join(map(str, range(n)))


def conj(xs):
    return xs[0] if len(xs) == 1 else '(' + ' && '.join(xs) + ')'


# Progress toward the goal, declared next to it: goalProgress(S) is in 0 .. goalMax and equals
# goalMax exactly in the goal states. A search that is being watched reports the best value it
# has seen; it is not part of any observation.
PROGRESS = '''
  --- progress toward the goal: 0 .. goalMax, goalMax exactly in the goal states
  op goalProgress : State -> Nat .
  op goalMax : -> Nat .
  op cnt : Bool Nat -> Nat .
  var GP-S : State .  var GP-N : Nat .
  eq cnt(true, GP-N) = s GP-N .
  eq cnt(false, GP-N) = GP-N .
'''


def count(props):
    """the number of the propositions that hold in GP-S, as a term"""
    t = '0'
    for q in reversed(props):
        t = f'cnt(GP-S |= {q}, {t})'
    return t


def steps(cases, last='0'):
    """nested if: the value of the first case whose condition holds"""
    t = last
    for cond, val in reversed(cases):
        t = f'if {cond} then {val} else {t} fi'
    return t


def progress(kmax, term, extra=''):
    return PROGRESS + extra + f'  eq goalMax = {kmax} .\n  eq goalProgress(GP-S) = {term} .\n'


def pairs_prop(name, locs):
    """the proposition `name`: at least two of the store locations hold 1"""
    eqs = [f'  eq ([ store(({a} |-> # 1, {b} |-> # 1, STR)) CFG ]) |= {name} = true .'
           for i, a in enumerate(locs) for b in locs[i + 1:]]
    return f'  op {name} : -> Prop [ctor] .\n' + '\n'.join(eqs) + f'\n  eq S |= {name} = false [owise] .\n'


def variant(text, test, name, goal, prog):
    """the analysis file with a copy of its TEST module under another goal and progress measure"""
    i = text.index(f'mod {test} is')
    j = text.index('\nendm', i) + len('\nendm\n')
    blk = text[i:j].replace(f'mod {test} is', f'mod {name} is', 1)
    a = blk.index('  eq goal = ')
    b = blk.index('\n  op env')
    return text + '\n' + blk[:a] + f'  eq goal = {goal} .\n' + prog + blk[b:]


QUEUED = '''  op queued : State -> Bool .
  eq queued([ network((0 |-> mq(GP-I:Int, (GP-M:Msg, GP-ML:MsgList)), GP-NET:Map{Nat, PROMELA-Queue})) GP-CFG:KConfig ]) = true .
  eq queued(GP-S) = false [owise] .
'''


def qlock(n):
    locs = ', '.join(f"(lval('in{i}, 0) |-> loc({i}))" for i in range(n))
    goal = '\n'.join(f'  eq ([ store(({i} |-> # 1, STR)) CFG ]) |= someInCS = true .' for i in range(n))
    plain = ' ; '.join([f"varEq('in{i}, {i}, 1)" for i in range(n)] + ['allDone', 'someInCS'])
    pid = ' ; '.join([f"assignP(NQ, 'in{i})" for i in range(n)] + ['sendP(NQ, 0)', 'recvP(NQ, 0)', 'locked(NQ)'])
    bodies = '\n'.join(
        f"  op body{i} : -> Seq .\n  eq body{i} = ( ('q ! # {i}) ; ('q ?[ # {i} ]) ;\n"
        f"               ('in{i} = # 1) ; ('in{i} = # 0) ; ('q ? # {i}) ; .Decls ) ." for i in range(n))
    store = ', '.join(f'({i} |-> # 0)' for i in range(n))
    two = pairs_prop('twoInCS', list(range(n)))
    prog = progress(2, steps([('GP-S |= someInCS', 2), ('queued(GP-S)', 1)]), QUEUED)
    mutex = progress(2, steps([('GP-S |= twoInCS', 2), ('GP-S |= someInCS', 1)]))
    return variant(f'''*** RL analysis layer for the {n}-process PROMELA qlock (FIFO mutex) model.
*** Goal = someInCS. Loc layout: in<i> = i; queue on channel 0 (capacity {n}).

{HEAD}
mod MODEL is
  protecting PROMELA-AGCEL .
  subsort GlobalState < State .

  var S : State .
  var CFG : KConfig .
  var STR : Map{{Nat, PROMELA-Value}} .
  vars FV FV' : FeatVec .

  op someInCS : -> Prop [ctor] .
{goal}
  eq S |= someInCS = false [owise] .

  --- error condition (not observed): two processes in their critical sections at once
{two}
  op plainProps : -> PropList .
  eq plainProps = {plain} .

  var NQ : Nat? .
  eq pidProps(NQ) =
    {pid} .

  eq reward(< FV , (someInCS : true) , FV' >) = 1.0 .
  eq reward(MS:MDPState) = 0.0 [owise] .
endm
{OBS}
mod QLOCK-PROMELA-TEST is
  pr MODEL .
  op goal : -> Prop .
  eq goal = someInCS .
{prog}
  op env : -> Map{{PROMELA-LVal, PROMELA-Ref}} .
  eq env = ((lval('q, 0) |-> ch(0)), {locs}) .
{envs(n)}

{bodies}

  op init : -> GlobalState .
  eq init = [
    proctypes(none)
    procs(
{procs(n)} )
    store(({store}))
    network((0 |-> mq({n}, nil)))
    lock(#none) active({ids(n)})
    nextLoc({n}) nextPid({n}) nextBCid(1) nextHCid(0) ] .
endm
''', 'QLOCK-PROMELA-TEST', 'QLOCK-PROMELA-MUTEX', 'twoInCS', mutex)


def dining(n):
    env = ', '.join([f"(lval('fork{i}, 0) |-> loc({i}))" for i in range(n)]
                    + [f"(lval('eat{i}, 0) |-> loc({n + i}))" for i in range(n)])
    preds = []
    for i in range(n):
        preds.append(f'''  eq ([ procs(proc(pid({i}) INNER-REST k(.K)) PR) CFG ]) |= p{i}Done = true .
  eq S |= p{i}Done = false [owise] .
''')
    for i in range(n):
        preds.append(f'''  eq ([ store(({n + i} |-> # 1, STR)) CFG ]) |= eat{i} = true .
  eq S |= eat{i} = false [owise] .
''')
    for i in range(n):
        preds.append(f'''  ceq ([ procs(proc(pid({i}) INNER-REST k((@[ G ~> KL ] BR) ~> KG) env(E)) PR)
         store(ST) network(NET) CFG ]) |= p{i}Blocked = true
    if G :: Expr /\\ isExec(G, E, ST, NET, {i}) = false .
  eq S |= p{i}Blocked = false [owise] .
''')
    blocked = ' /\\ '.join(f'(GS |= p{i}Blocked) = true' for i in range(n))
    names = ' '.join([f'p{i}Done' for i in range(n)] + [f'p{i}Blocked' for i in range(n)] + ['lockFree']
                     + [f'eat{i}' for i in range(n)])
    plain = ' ; '.join([f'p{i}Blocked' for i in range(n)] + [f'p{i}Done' for i in range(n)] + ['lockFree']
                       + [f'eat{i}' for i in range(n)] + ['deadlock'])
    pid = ' ; '.join([f"assignP(NQ, 'fork{i})" for i in range(n)] + [f"assignP(NQ, 'eat{i})" for i in range(n)]
                     + ['locked(NQ)'])
    bodies = []
    for i in range(n):
        a, b = i, (i + 1) % n
        bodies.append(f'''  op body{i} : -> Seq .
  eq body{i} = ( atomic{{ ('fork{a} == # 0) ; ('fork{a} = # 1) ; .Decls }} ;
               atomic{{ ('fork{b} == # 0) ; ('fork{b} = # 1) ; .Decls }} ;
               ('eat{i} = # 1) ; ('eat{i} = # 0) ;
               ('fork{a} = # 0) ; ('fork{b} = # 0) ; .Decls ) .''')
    store = ', '.join(f'({i} |-> # 0)' for i in range(2 * n))
    return f'''*** RL analysis layer for the {n}-process PROMELA dining philosophers model.
*** Philosopher i takes fork i, then fork (i+1) mod {n}. Goal = deadlock (every process blocked).
*** Loc layout: fork<i> = i, eat<i> = {n} + i.

{HEAD}
mod MODEL is
  protecting PROMELA-AGCEL .
  subsort GlobalState < State .

  var S : State .
  var GS : GlobalState .
  vars CFG PR INNER-REST : KConfig .
  vars ST STR : Map{{Nat, PROMELA-Value}} .
  var NET : Map{{Nat, PROMELA-Queue}} .
  var E : Map{{PROMELA-LVal, PROMELA-Ref}} .
  var G : BasicStmt .
  vars KL KG : K .
  var BR : BranchSet .
  vars FV FV' : FeatVec .

  op deadlock : -> Prop [ctor] .
  ops {names} : -> Prop [ctor] .

{''.join(preds)}
  eq ([ lock(#none) CFG ]) |= lockFree = true .
  eq S |= lockFree = false [owise] .

  op canProgress : GlobalState -> Bool .
  ceq canProgress(GS) = false
    if {blocked} .
  eq canProgress(GS) = true [owise] .

  ceq S |= deadlock = true if canProgress(S) = false .
  eq S |= deadlock = false [owise] .

  op plainProps : -> PropList .
  eq plainProps = {plain} .

  var NQ : Nat? .
  eq pidProps(NQ) =
    {pid} .

  eq reward(< FV , (deadlock : true) , FV' >) = 1.0 .
  eq reward(MS:MDPState) = 0.0 [owise] .
endm
{OBS}
mod DINING-PROMELA-TEST is
  pr MODEL .
  op goal : -> Prop .
  eq goal = deadlock .
{progress(n, count([f'p{i}Blocked' for i in range(n)]))}
  op env : -> Map{{PROMELA-LVal, PROMELA-Ref}} .
  eq env = ({env}) .
{envs(n)}

{chr(10).join(bodies)}

  op init : -> GlobalState .
  eq init = [
    proctypes(none)
    procs(
{procs(n)} )
    store(({store}))
    network(empty) lock(#none) active({ids(n)})
    nextLoc({2 * n}) nextPid({n}) nextBCid(0) nextHCid(0) ] .
endm
'''


def filt(n):
    lv = {f'level{i}': i for i in range(n)}
    vi = {f'victim{L}': n + L - 1 for L in range(1, n)}
    inn = {f'in{i}': 2 * n - 1 + i for i in range(n)}
    allv = {**lv, **vi, **inn}
    env = ', '.join(f"(lval('{k}, 0) |-> loc({v}))" for k, v in allv.items())
    goal = '\n'.join(f'  eq ([ store(({v} |-> # 1, STR)) CFG ]) |= someInCS = true .' for v in inn.values())
    plain = ' ; '.join([f"varEq('{k}, {v}, {x})" for k, v in {**lv, **vi}.items() for x in range(1, n)]
                       + [f"varEq('{k}, {v}, 1)" for k, v in inn.items()] + ['allDone', 'someInCS'])
    pid = ' ; '.join([f"assignP(NQ, '{k})" for k in allv] + ['locked(NQ)'])
    bodies = []
    for i in range(n):
        st = []
        for L in range(1, n):
            below = conj([f"('level{j} < # {L})" for j in range(n) if j != i])
            st.append(f"('level{i} = # {L}) ; ('victim{L} = # {i}) ;\n"
                      f"               ({below} || ('victim{L} != # {i})) ;")
        st.append(f"('in{i} = # 1) ; ('in{i} = # 0) ; ('level{i} = # 0) ; .Decls")
        bodies.append(f'  op body{i} : -> Seq .\n  eq body{i} = ( ' + '\n               '.join(st) + ' ) .')
    store = ', '.join(f'({i} |-> # 0)' for i in range(len(allv)))
    two = pairs_prop('twoInCS', list(inn.values()))
    at = lambda L: ' or '.join(f"(GP-S |= varEq('{k}, {v}, {L}))" for k, v in lv.items())
    prog = progress(n, steps([('GP-S |= someInCS', n)] + [(at(L), L) for L in range(n - 1, 0, -1)]))
    mutex = progress(2, steps([('GP-S |= twoInCS', 2), ('GP-S |= someInCS', 1)]))
    return variant(f'''*** RL analysis layer for the {n}-process PROMELA filter lock model ({n - 1} level(s)).
*** Process i at level L: level<i> = L ; victim<L> = i ; wait until every other process is
*** below level L or victim<L> != i. Goal = someInCS.
*** Loc layout: level<i> = i, victim<L> = {n} + L - 1, in<i> = {2 * n - 1} + i.

{HEAD}
mod MODEL is
  protecting PROMELA-AGCEL .
  subsort GlobalState < State .

  var S : State .
  var CFG : KConfig .
  var STR : Map{{Nat, PROMELA-Value}} .
  vars FV FV' : FeatVec .

  op someInCS : -> Prop [ctor] .
{goal}
  eq S |= someInCS = false [owise] .

  --- error condition (not observed): two processes in their critical sections at once
{two}
  op plainProps : -> PropList .
  eq plainProps = {plain} .

  var NQ : Nat? .
  eq pidProps(NQ) =
    {pid} .

  eq reward(< FV , (someInCS : true) , FV' >) = 1.0 .
  eq reward(MS:MDPState) = 0.0 [owise] .
endm
{OBS}
mod FILTER-PROMELA-TEST is
  pr MODEL .
  op goal : -> Prop .
  eq goal = someInCS .
{prog}
  op env : -> Map{{PROMELA-LVal, PROMELA-Ref}} .
  eq env = ({env}) .
{envs(n)}

{chr(10).join(bodies)}

  op init : -> GlobalState .
  eq init = [
    proctypes(none)
    procs(
{procs(n)} )
    store(({store}))
    network(empty) lock(#none) active({ids(n)})
    nextLoc({len(allv)}) nextPid({n}) nextBCid(0) nextHCid(0) ] .
endm
''', 'FILTER-PROMELA-TEST', 'FILTER-PROMELA-MUTEX', 'twoInCS', mutex)


def onethirdrule(n):
    names = ([f'val{i}' for i in range(n)] + [f'sent{i}' for i in range(n)] + [f'arr{i}' for i in range(n)]
             + [f'decided{i}' for i in range(n)])
    loc = {k: j for j, k in enumerate(names)}
    env = ', '.join(f"(lval('{k}, 0) |-> loc({v}))" for k, v in loc.items())
    alld = ', '.join(f"{loc[f'decided{i}']} |-> # 1" for i in range(n))
    plain = ' ; '.join([f"varEq('arr{i}, {loc[f'arr{i}']}, 1)" for i in range(n)]
                       + [f"varEq('decided{i}, {loc[f'decided{i}']}, 1)" for i in range(n)]
                       + ['allDone', 'allDecided'])
    pid = ' ; '.join([f"assignP(NQ, 'sent{i})" for i in range(n)] + [f"assignP(NQ, 'arr{i})" for i in range(n)]
                     + [f"assignP(NQ, 'val{i})" for i in range(n)]
                     + [f"assignP(NQ, 'decided{i})" for i in range(n)] + ['locked(NQ)'])
    bodies = []
    for i in range(n):
        arrived = conj([f"('arr{j} == # 1)" for j in range(n)])
        opts = []
        for k in range(n):  # sent<k> is the minimum; the lowest index wins ties
            g = conj([f"('sent{k} <= 'sent{j})" if j > k else f"('sent{k} < 'sent{j})" for j in range(n) if j != k])
            opts.append(f"(:: {g} ; ('val{i} = 'sent{k}) ; .Decls)")
        bodies.append(f'''  op body{i} : -> Seq .
  eq body{i} = ( ('sent{i} = 'val{i}) ; ('arr{i} = # 1) ;
               {arrived} ;
               (if {(chr(10) + '                   ').join(opts)} fi) ;
               ('decided{i} = # 1) ; .Decls ) .''')
    store = ', '.join(f'({loc[k]} |-> # {int(k[3:]) if k.startswith("val") else 0})' for k in names)
    return f'''*** RL analysis layer for the {n}-process PROMELA One-Third-Rule consensus model (one round).
*** Goal = allDecided. Initial values 0..{n - 1}.
*** Loc layout: val<i> = i, sent<i> = {n} + i, arr<i> = {2 * n} + i, decided<i> = {3 * n} + i.

{HEAD}
mod MODEL is
  protecting PROMELA-AGCEL .
  subsort GlobalState < State .

  var S : State .
  var CFG : KConfig .
  var STR : Map{{Nat, PROMELA-Value}} .
  vars FV FV' : FeatVec .

  op allDecided : -> Prop [ctor] .
  eq ([ store(({alld}, STR)) CFG ]) |= allDecided = true .
  eq S |= allDecided = false [owise] .

  op plainProps : -> PropList .
  eq plainProps = {plain} .

  var NQ : Nat? .
  eq pidProps(NQ) =
    {pid} .

  eq reward(< FV , (allDecided : true) , FV' >) = 1.0 .
  eq reward(MS:MDPState) = 0.0 [owise] .
endm
{OBS}
mod OTR-PROMELA-TEST is
  pr MODEL .
  op goal : -> Prop .
  eq goal = allDecided .
{progress(n, count([f"varEq('decided{i}, {loc[f'decided{i}']}, 1)" for i in range(n)]))}
  op env : -> Map{{PROMELA-LVal, PROMELA-Ref}} .
  eq env = ({env}) .
{envs(n)}

{chr(10).join(bodies)}

  op init : -> GlobalState .
  eq init = [
    proctypes(none)
    procs(
{procs(n)} )
    store(({store}))
    network(empty) lock(#none) active({ids(n)})
    nextLoc({len(names)}) nextPid({n}) nextBCid(0) nextHCid(0) ] .
endm
'''

# The listing of exercise 4.c as published, with the page's line numbers.
EX4C_URL = 'http://spinroot.com/spin/Man/Exercises.html'
EX4C = """\
  1 byte in;                                    /* file ex.4c */
  2 byte x, y, z;
  3 active [2] proctype user()
  4 {	byte me = _pid+1;			/* me is 1 or 2 */
  5 L1:	x = me;
  6 L2:	if
  7 	:: (y != 0 && y != me) -> goto L1	/* try again */
  8 	:: (y == 0 || y == me)
  9 	fi;
 10 L3:	z = me;
 11 L4:	if
 12 	:: (x != me)  -> goto L1		/* try again */
 13 	:: (x == me)
 14 	fi;
 15 L5:	y = me;
 16 L6:	if
 17 	:: (z != me) -> goto L1			/* try again */
 18 	:: (z == me)
 19 	fi;
 20 L7:						/* success */
 21 	in = in+1;
 22 	assert(in == 1);
 23 	in = in - 1;
 24 	goto L1
 25 }
"""


def exfourc(n):
    vs = {'cnt': 0, 'x': 1, 'y': 2, 'z': 3}   # in declaration order; me of process i is local, at 4 + i
    env = ', '.join(f"(lval('{k}, 0) |-> loc({v}))" for k, v in vs.items())
    ins = '\n'.join(f'  eq ([ store((0 |-> # {v}, STR)) CFG ]) |= someInCS = true .' for v in range(1, n + 1))
    two = '\n'.join(f'  eq ([ store((0 |-> # {v}, STR)) CFG ]) |= twoInCS = true .' for v in range(2, n + 1))
    plain = ' ; '.join([f"varEq('{k}, {v}, {x})" for k, v in vs.items() for x in range(1, n + 1)] + ['allDone', 'twoInCS'])
    pid = ' ; '.join([f"assignP(NQ, '{k})" for k in vs] + ['locked(NQ)'])
    envs_ = '\n'.join(f"  op env{i} : -> Map{{PROMELA-LVal, PROMELA-Ref}} .\n  eq env{i} = (env, (lval('me, 0) |-> loc({4 + i}))) ."
                      for i in range(n))
    procs_ = '\n'.join(f"      proc(pid({i}) name('user) k( body ~> .K ) local('me) env(env{i}))" for i in range(n))
    store = ', '.join([f'({v} |-> # 0)' for v in vs.values()] + [f'({4 + i} |-> # {i + 1})' for i in range(n)])
    listing = '\n'.join('*** ' + l for l in EX4C.splitlines())
    mutex = progress(2, steps([('GP-S |= twoInCS', 2), ('GP-S |= someInCS', 1)]))
    text = f'''*** RL analysis layer for the {n}-process PROMELA model ex.4c, a faulty mutual exclusion algorithm.
*** Source: G. J. Holzmann, "Spin Verification Examples and Exercises", exercise 4.c,
*** {EX4C_URL},
*** the listing with the comment "/* file ex.4c */". The page presents it as a mutual exclusion
*** algorithm a computer manufacturer recommended to a customer, one of the faulty solutions.
*** The listing as published, with the page's line numbers:
***
{listing}
***
*** Only the number of processes was generalised (active [2] becomes N = {n} processes); apart from
*** the adjustments below, which the backend requires, nothing in the algorithm was changed.
***   A1  _pid is not supported: process i (pid i, 0 .. {n - 1}) has its own local me = i + 1 (line 4,
***       "byte me = _pid+1"), given through its own env and store entry. active [N] would declare
***       me with its default value 0 in every process.
***   A2  in is a keyword since Spin 6 (Examples/Book_1991/README.txt of the Spin distribution): it is cnt.
***   A3  assert is not a statement of the backend: line 22, "assert(in == 1);", is the goal twoInCS,
***       cnt >= 2. cnt counts the processes between the increment and the decrement, so the
***       assertion can fail exactly when a state with cnt >= 2 is reachable.
***   C2  the backend has no labelled statement and builds no goto map: the retry loop is a do loop
***       with the same guards. A failed guard ends the option and the loop starts again at x = me,
***       which is "goto L1" of lines 7, 12, 17; the end of the success branch is "goto L1" of line 24.
***   byte is not a type of the backend: the values stay within 0 .. {n}, far below 256, and the
***       store holds them as integers.
*** The decrement of line 23 is written with the prefix form of _-_, the same term: the mixfix form
*** 'cnt - # 1 also parses as 'cnt juxtaposed with - # 1.
*** Goal = twoInCS. Loc layout: cnt = 0, x = 1, y = 2, z = 3, me of process i = 4 + i.

{HEAD}
mod MODEL is
  protecting PROMELA-AGCEL .
  subsort GlobalState < State .

  var S : State .
  var CFG : KConfig .
  var STR : Map{{Nat, PROMELA-Value}} .
  vars FV FV' : FeatVec .

  --- processes in the critical section, between the increment and the decrement of cnt
  op someInCS : -> Prop [ctor] .
{ins}
  eq S |= someInCS = false [owise] .

  --- the goal: two processes in their critical sections at once (the assertion of line 22 fails)
  op twoInCS : -> Prop [ctor] .
{two}
  eq S |= twoInCS = false [owise] .

  op plainProps : -> PropList .
  eq plainProps = {plain} .

  var NQ : Nat? .
  eq pidProps(NQ) =
    {pid} .

  eq reward(< FV , (twoInCS : true) , FV' >) = 1.0 .
  eq reward(MS:MDPState) = 0.0 [owise] .
endm
{OBS}
mod EXFOURC-PROMELA-TEST is
  pr MODEL .
  op goal : -> Prop .
  eq goal = twoInCS .
{mutex}
  op env : -> Map{{PROMELA-LVal, PROMELA-Ref}} .
  eq env = ({env}) .
{envs_}

  --- the body of proctype user, lines 5-24, with the retry loop as a do loop (C2 above)
  op body : -> Seq .
  eq body = (
    (do (:: ('x = 'me) ;                                       --- line 5   L1: x = me
          (if (:: (('y != # 0) && ('y != 'me)) ; .Decls)      --- line 7   then goto L1
              (:: (('y == # 0) || ('y == 'me)) ;              --- line 8
                  ('z = 'me) ;                                 --- line 10  L3: z = me
                  (if (:: ('x != 'me) ; .Decls)                --- line 12  then goto L1
                      (:: ('x == 'me) ;                        --- line 13
                          ('y = 'me) ;                         --- line 15  L5: y = me
                          (if (:: ('z != 'me) ; .Decls)        --- line 17  then goto L1
                              (:: ('z == 'me) ;                --- line 18
                                  ('cnt = 'cnt + # 1) ;        --- line 21  L7: in = in+1
                                  ('cnt = _-_('cnt, # 1)) ;    --- line 23  in = in - 1, then goto L1
                                  .Decls)
                           fi) ; .Decls)
                   fi) ; .Decls)
           fi) ; .Decls)
     od) ; .Decls ) .

  op init : -> GlobalState .
  eq init = [
    proctypes(none)
    procs(
{procs_} )
    store(({store}))
    network(empty) lock(#none) active({ids(n)})
    nextLoc({4 + n}) nextPid({n}) nextBCid(0) nextHCid(0) ] .
endm
'''
    for line in text.splitlines():   # a comment line whose text starts with "(" opens a bracket comment
        for mark in ('***', '---'):
            if mark in line:
                assert not line.split(mark, 1)[1].lstrip().startswith('('), line
    return text


GEN = {'qlock': qlock, 'dining': dining, 'filter': filt, 'onethirdrule': onethirdrule, 'exfourc': exfourc}


def generate(family, n):
    if family not in GEN:
        raise ValueError(f'unknown family {family!r}; one of {sorted(GEN)}')
    if n < 2:
        raise ValueError('n must be at least 2')
    return GEN[family](n)


if __name__ == '__main__':
    fam, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    with open(out, 'w') as f:
        f.write(generate(fam, n))
