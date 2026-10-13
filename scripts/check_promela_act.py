#!/usr/bin/env python3
"""Regression for the generated parts of the PROMELA semantics: canStep, the condition of
[loose], and the action-recording semantics (PROMELA-SEMANTICS-ACT).

  sync         the canStep block of semantics/promela-semantics.maude and
               semantics/promela-semantics-act.maude are what gen_promela_act.py generates;
               canStep has one equation per rule other than [loose];
               semantics/promela-obs-generated.maude is what promela-rule-extract.maude prints
  sync --write-hook   the same, after rewriting semantics/promela-obs-generated.maude
  prefix       prefix(K, N) on a hand-built computation, N = 0, 1, 2 and larger than K
  tests        tests / tests-extended / tests-all / tests-init give, over PROMELA-SEMANTICS-ACT,
               the results they give over PROMELA-SEMANTICS once last(...) is removed
  safety       the six safety files: every search has the same verdict over both modules; the
               state counts over PROMELA-SEMANTICS-ACT are printed for propKsize 0, 1, 2
  transitions  from the initial states of the safety files, of the test files and of the
               atomicity files: for every reachable configuration and every rule label, the
               successors over PROMELA-SEMANTICS-ACT without last(...) are the successors over
               PROMELA-SEMANTICS, and every successor carries
               last(exec(label, tag(pid), prefix(computation of that pid before the step, propKsize)));
               over both modules canStep is true exactly in the configurations a rule other
               than [loose] can rewrite, and [loose] applies only where canStep is false
  atomic       semantics/tests-atomic.maude gives the verdicts of its Expect lines, and the
               examples that depend on atomic blocks give theirs, over both modules
  props        on the dining philosophers analysis layer: execP / execKP are false initially,
               hold after the matching step, and are listed in obs(init)

  check_promela_act.py [check ...]        all checks when none is named; exit 1 on a failure
"""
import ast, json, os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_promela_act as gen

REPO, SEM = gen.REPO, gen.SEM
BP = os.path.join(REPO, 'benchmarks-promela')
MAUDE = os.environ.get('MAUDE', '/usr/local/bin/maude')
SAFETY = ['dining-philosophers-2', 'filter-2procs', 'qlock-2procs', 'onethirdrule-2procs', 'chan-buffered-1', 'bakery-2procs']
TESTS = ['tests', 'tests-extended', 'tests-all', 'tests-init']
# files whose verdicts depend on atomic blocks: (path, has each search a solution; None: the
# Expect lines of the file), and for the transition check (module, start terms)
ATOMIC = [
    ('semantics/tests-atomic.maude', None, 'ATOMIC-TESTS', ['initA1', 'initA2', 'initA3']),
    ('benchmarks-promela/spink-examples/spin-atomic1.maude', [False], 'BENCH-SPIN-ATOMIC1', ['init']),
    ('benchmarks-promela/spink-examples/spin-atomic2.maude', [False], 'BENCH-SPIN-ATOMIC2', ['init']),
    ('benchmarks-promela/spink-examples/paper-spin-atomic-handshake.maude', [False], 'BENCH-PSAH', ['init']),
    ('benchmarks-promela/bakery-2procs.maude', [False, True], None, []),   # mutexViolated, someInCrit
]
COMMAND = re.compile(r'^(?:rew|frew|search|red)\b.*?\s\.[ \t]*$|^quit[ \t]*$', re.M | re.S)
LIMIT = 20000  # configurations explored per initial state in the transition check
# start terms for the rules that neither the benchmarks nor the test files exercise
EXTRA = [
    "initT(#globals((bool 'b) ; (bool 'c[# 2]) ; (int 'a[# 2]) ; (int 'y = # 5) ; (chan 'u[# 1]) ; .Decls)"
    " ~> #activeN(2, 'P, .Decls, (('a[# 0] = # 1) ; ('b = # true) ; .Decls)))",
    "initT(#globals((int 'x[# 1] = # 0) ; .Decls) ~> #proctype('Q, (int 'p ; .Decls), (('x = 'p) ; .Decls))"
    " ~> #run('Q, # 7))",
]


def maude(path, cwd):
    env = dict(os.environ, MAUDE_LIB=SEM)
    r = subprocess.run([MAUDE, '-no-banner', '-batch', path], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True)
    return r.stdout, r.stderr


def blocks(out):
    """the output of each command"""
    out = re.sub(r'^Bye\.\s*\Z', '', out, flags=re.M)
    return [b for b in re.split(r'^={10,}$', out, flags=re.M) if b.strip()]


def strip_last(s):
    while True:
        i = s.find('last(')
        if i < 0:
            return s
        s = s[:i] + s[gen.balanced(s, i + 4, '(', ')'):]


def results(out):
    """per command: for a search whether it has a solution, otherwise the result without
    last(...); rewrite counts, state numbers and whitespace are dropped"""
    res = []
    for b in blocks(out):
        b = re.sub(r'rewrites: \d+ in \d+ms cpu \(\d+ms real\) \([\d~]+\s+rewrites/second\)', '', b)
        if b.lstrip().startswith('search'):
            res.append('search: ' + ('solution' if re.search(r'^Solution 1 ', b, re.M) else 'no solution'))
        else:
            res.append(re.sub(r'\s+', '', strip_last(b[b.find('result'):] if 'result' in b else b)))
    return res


def act_test(text):
    """a test file over PROMELA-SEMANTICS-ACT: explicit start configurations get last(noAct)"""
    text = text.replace('load promela-semantics.maude .', f'load {gen.DST} .\n'
                        'mod PROMELA-SEMANTICS-ACT-T is\n  protecting PROMELA-SEMANTICS-ACT .\n'
                        '  eq focusIds = mt .\n  eq propKsize = 0 .\nendm')
    def cmd(m):
        c = m.group(0).replace('in PROMELA-SEMANTICS :', 'in PROMELA-SEMANTICS-ACT-T :')
        c = re.sub(r'(in PROMELA-SEMANTICS-ACT-T :\s*\[)', r'\1 last(noAct) ', c, count=1)
        # a search pattern that spells out the whole configuration must allow the action
        return re.sub(r'(=>[*!+1]\s*\[)', r'\1 last(LA:Action) ', c, count=1)
    text = re.sub(r'^(?:rew|frew|search|red)\b.*?\s\.[ \t]*$', cmd, text, flags=re.M | re.S)
    return re.sub(r'\bprotecting PROMELA-SEMANTICS \.', 'protecting PROMELA-SEMANTICS-ACT-T .', text)


HOOK = os.path.join(SEM, 'promela-obs-generated.maude')


def hook_text():
    """the module promela-rule-extract.maude prints for the rules of PROMELA-SEMANTICS"""
    r = subprocess.run([MAUDE, '-no-banner', '-no-advise', '-no-wrap', '-batch', 'promela-rule-extract.maude'], cwd=SEM,
                       env=dict(os.environ, MAUDE_LIB=SEM), stdin=subprocess.DEVNULL, capture_output=True, text=True)
    out = r.stdout
    lit = re.sub(r'\s*Bye\.\s*\Z', '', out[out.rindex('result String:') + len('result String:'):]).strip()
    return ast.literal_eval(lit)


def check_sync(write_hook=False):
    hook = hook_text()
    if write_hook and open(HOOK).read() != hook:
        with open(HOOK, 'w') as f:
            f.write(hook)
        print(f'sync: rewrote {HOOK}')
    hook_ok = open(HOOK).read() == hook
    src = open(gen.SRC).read()
    block, eqs = gen.canstep(src[:src.index(gen.CS_BEGIN)] + src[src.index(gen.CS_END) + len(gen.CS_END):])
    new, info = gen.semantics(src)
    n = len(re.findall(r'^[ \t]*c?eq canStep\(', src, re.M))
    ok = (gen.with_canstep(src) == src and os.path.exists(gen.DST) and open(gen.DST).read() == new
          and len(eqs) == len(info) - len(gen.OWISE) and n == len(eqs) + 1 and hook_ok)
    print(f'sync: {"ok" if ok else "FAIL"} ({len(info)} rules; canStep: {len(eqs)} equations, one per rule other than '
          f'{", ".join("[" + l + "]" for l in gen.OWISE)}, and the owise equation; the action-binding hook is '
          f'{"what" if hook_ok else "NOT what"} promela-rule-extract.maude prints)')
    return ok


def check_prefix(tmp):
    k = "#rel ~> (@[ # true ~> #rel ] @[ # false ~> .K ]) ~> #wait"
    want = {0: '.K', 1: '#rel', 2: '#rel ~> (@[ # true ~> #rel ] @[ # false ~> .K ])', 3: k, 9: k}
    p = os.path.join(tmp, 'prefix.maude')
    with open(p, 'w') as f:
        f.write(f'load {gen.DST} .\n')
        for n, w in want.items():
            f.write(f'red in PROMELA-SEMANTICS-ACT : prefix({k}, {n}) .\nred in PROMELA-SEMANTICS-ACT : {w} .\n')
    r = results(maude(p, tmp)[0])
    got = [r[2 * i] == r[2 * i + 1] for i in range(len(want))] if len(r) == 2 * len(want) else []
    ok = got == [True] * len(want)
    print(f'prefix: {"ok" if ok else "FAIL"} (sizes {sorted(want)}: ' + ' '.join(x[len('resultK:'):] for x in r[::2]) + ')')
    return ok


def check_tests(tmp):
    ok = True
    for t in TESTS:
        src = os.path.join(SEM, t + '.maude')
        base = results(maude(src, SEM)[0])
        p = os.path.join(tmp, t + '-act.maude')
        with open(p, 'w') as f:
            f.write(act_test(open(src).read()))
        act = results(maude(p, SEM)[0])
        same = base == act
        ok &= same
        print(f'tests: {t}: {len(base)} results, {"same over both modules" if same else "DIFFERENT"}')
        if not same:
            for i, (x, y) in enumerate(zip(base, act)):
                if x != y:
                    print(f'  first difference at result {i + 1}:\n    {x[:300]}\n    {y[:300]}'); break
    return ok


def verdicts(out):
    """per search: whether it has a solution, and the number of states it reports last"""
    v = []
    for b in blocks(out):
        if not b.lstrip().startswith('search'): continue
        n = re.findall(r'states: (\d+)', b)
        v.append(('Solution 1 ' in b, int(n[-1]) if n else None))
    return v


def check_safety(tmp):
    ok = True
    print('safety: file | searches: solution found (states) over PROMELA-SEMANTICS | over PROMELA-SEMANTICS-ACT with propKsize 0 / 1 / 2 (no focus)')
    for b in SAFETY:
        src = os.path.join(BP, b + '.maude')
        base = verdicts(maude(src, BP)[0])
        row = []
        for ks in (0, 1, 2):
            p = os.path.join(tmp, f'{b}-act-k{ks}.maude')
            with open(p, 'w') as f:
                f.write(gen.act_file(open(src).read(), BP, focus=[], ksize=ks))
            act = verdicts(maude(p, tmp)[0])
            same = [x[0] for x in act] == [x[0] for x in base] and len(act) == len(base)
            ok &= same
            row.append(('' if same else 'VERDICT DIFFERS ') + ', '.join(f'{"yes" if s else "no"} ({n})' for s, n in act))
        print(f'  {b} | ' + ', '.join(f'{"yes" if s else "no"} ({n})' for s, n in base) + ' | ' + ' / '.join(row))
    print(f'safety: {"ok" if ok else "FAIL"}')
    return ok


def start_terms():
    """[(name, file or None, module, start term)] for the transition check"""
    out = []
    for b in SAFETY:
        text = open(os.path.join(BP, b + '.maude')).read()
        mod = re.findall(r'^mod (\S+) is', text, re.M)[-1]
        out.append((b, os.path.join(BP, b + '.maude'), mod, 'init'))
    for rel, _, mod, starts in ATOMIC:
        for st in starts:
            out.append((os.path.basename(rel)[:-len('.maude')] + (f':{st}' if len(starts) > 1 else ''),
                        os.path.join(REPO, rel), mod, st))
    for t in ('tests-all', 'tests-init'):
        text = open(os.path.join(SEM, t + '.maude')).read()
        k = 0
        for m in re.finditer(r'^(?:rew|frew|search)[ \t]*(?:\[\d+\])?[ \t]*(?:in PROMELA-SEMANTICS :)?\s*((?:\[|initT\().*?)(?:=>[*!+1]|\.\s*$)',
                             text, re.M | re.S):
            term = m.group(1).strip()
            if term.endswith(' .'): term = term[:-2]
            k += 1
            out.append((f'{t}#{k}', None, 'PROMELA-SEMANTICS', term))
    for k, term in enumerate(EXTRA, 1):
        out.append((f'extra#{k}', None, 'PROMELA-SEMANTICS', term))
    return out


WORKER = r'''
import json, sys, maude
spec = json.load(open(sys.argv[1]))
maude.init(advise=False)
for f in spec['load']: maude.load(f)
m = maude.getModule(spec['module'])
labels = sorted({r.getLabel() for r in m.getRules() if r.getLabel()})
act = spec['act']
def red(s):
    t = m.parseTerm(s)
    if t is None: raise SystemExit('cannot parse: ' + s[:200])
    t.reduce(); return t
def canon(s): return red(s).prettyPrint(0)
def proj(t): return red('noLast(' + t.prettyPrint(0) + ')').prettyPrint(0) if act else t.prettyPrint(0)
t0 = red(spec['start'])
seen, order, trans, bad, used = {proj(t0): t0}, [proj(t0)], {}, [], set()
ncan, badcan = 0, []
i = 0
while i < len(order) and len(order) <= spec['limit']:
    key = order[i]; t = seen[key]; i += 1
    cur = t.prettyPrint(0)
    for L in labels:
        succ = set()
        for rhs, sb, _, _ in t.apply(L):
            rhs.reduce()
            k = proj(rhs)
            succ.add(k)
            if k not in seen: seen[k] = rhs; order.append(k)
            if act:
                used.add(L)
                var = spec['pidvar'][L]
                pid = [v.prettyPrint(0) for x, v in sb if x.getVarName() == var][0]
                kt = '.K' if spec['nokt'].get(L) else f'prefix(kOf({cur}, {pid}), propKsize)'
                ok = red(f'lastOf({rhs.prettyPrint(0)}) == exec({L}, tag({pid}), {kt})').prettyPrint(0)
                if ok != 'true' and len(bad) < 5: bad.append((L, cur[:200], rhs.prettyPrint(0)[:300]))
        if succ: trans.setdefault(key, {})[L] = sorted(succ)
    steps = sorted(L for L in trans.get(key, {}) if L not in spec['owise'])
    # [loose] passes the configuration without last(...): a recorded computation may have no sort
    cs = red('canStep(' + ('noLast(' + cur + ')' if act else cur) + ')').prettyPrint(0)
    ncan += 1
    if cs != ('true' if steps else 'false') or (steps and any(L in trans.get(key, {}) for L in spec['owise'])):
        if len(badcan) < 5: badcan.append((cur[:300], cs, steps))
    seen[key] = None
same = None
if spec.get('base'):   # the transitions over PROMELA-SEMANTICS, printed again by this module
    memo = {}
    def c(s):
        if s not in memo: memo[s] = canon(s)
        return memo[s]
    base = json.load(open(spec['base']))
    same = {c(k): {L: sorted(map(c, v)) for L, v in d.items()} for k, d in base['trans'].items()} == trans \
        and base['states'] == len(order)
json.dump(dict(states=len(order), complete=i == len(order), trans=trans, bad=bad, used=sorted(used), same=same,
               ncan=ncan, badcan=badcan), open(sys.argv[2], 'w'))
'''

HELPER = '''
mod CHECK-ACT is
  protecting {mod} .
  eq focusIds = {focus} .
  eq propKsize = {ksize} .
  op kOf : [GlobalState] Int -> [K] .
  eq kOf([ procs(proc(pid(I:Int) INNER:[KConfig] k(K:[K])) PR:[KConfig]) CFG:[KConfig] ], I:Int) = K:[K] .
  op lastOf : [GlobalState] -> [Action] .
  eq lastOf([ last(LA:[Action]) CFG:[KConfig] ]) = LA:[Action] .
  op noLast : [GlobalState] -> [GlobalState] .
  eq noLast([ last(LA:[Action]) CFG:[KConfig] ]) = [ CFG:[KConfig] ] .
endm
'''


def check_transitions(tmp):
    _, info = gen.semantics(open(gen.SRC).read())
    pidvar = {l: p for l, p, _ in info}
    nokt = {l: k == '.K' for l, _, k in info}
    worker = os.path.join(tmp, 'worker.py')
    with open(worker, 'w') as f:
        f.write(WORKER)
    ok, used, n, ncan = True, set(), 0, 0
    for name, path, mod, start in start_terms():
        res = {}
        src_dir = os.path.dirname(path) if path else None
        absolute = lambda m: f'load {os.path.normpath(os.path.join(src_dir, m.group(1) + ".maude"))} .'
        for focus, ksize in ((None, None), ('mt', 0), ('0, 1', 2)):
            act = focus is not None
            stem = os.path.join(tmp, re.sub(r'\W', '_', f'tr_{name}_{focus}_{ksize}'))
            if path is None:
                load = [gen.DST if act else gen.SRC]
            elif act:
                load = [stem + '.maude']
                with open(load[0], 'w') as f:
                    f.write(COMMAND.sub('', gen.act_file(open(path).read(), src_dir)))
            else:
                load = [stem + '.maude']
                with open(load[0], 'w') as f:
                    f.write(gen.LOAD.sub(absolute, COMMAND.sub('', open(path).read())))
            module, st = mod, start
            if act:
                module = 'CHECK-ACT'
                amod = mod + '-ACT' if mod == 'PROMELA-SEMANTICS' else mod
                helper = stem + '_helper.maude'
                with open(helper, 'w') as f:
                    f.write(HELPER.format(mod=amod, focus=focus, ksize=ksize))
                load.append(helper)
                if st.startswith('['):
                    st = '[ last(noAct)' + st[1:]
            with open(stem + '.json', 'w') as f:
                json.dump(dict(load=load, module=module, start=st, act=act, limit=LIMIT, pidvar=pidvar, nokt=nokt,
                               owise=list(gen.OWISE), base=base_out if act else None), f)
            if not act:
                base_out = stem + '.out.json'
            r = subprocess.run([sys.executable, worker, stem + '.json', stem + '.out.json'],
                               env=dict(os.environ, MAUDE_LIB=SEM, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True)
            if r.returncode:
                print(f'transitions: {name}: worker failed: {(r.stdout + r.stderr)[-400:]}'); ok = False; res = None; break
            res[focus] = json.load(open(stem + '.out.json'))
        if res is None: continue
        base = res[None]
        for focus, a in res.items():
            ncan += a['ncan']
            if a['badcan']:
                ok = False
                print(f'transitions: {name} ({"PROMELA-SEMANTICS" if focus is None else "focus " + focus}): '
                      f'canStep DISAGREES with the rules, e.g. (configuration, canStep, rules that apply) = {a["badcan"][0]}')
        for focus in ('mt', '0, 1'):
            a = res[focus]
            used |= set(a['used'])
            same = a['same']
            if not same or a['bad']:
                ok = False
                print(f'transitions: {name} (focus {focus}): ' + ('DIFFERENT transitions ' if not same else '')
                      + (f'WRONG recorded action {a["bad"][:1]}' if a['bad'] else ''))
        n += 1
        if path is not None:
            print(f'transitions: {name}: {base["states"]} configurations' + ('' if base['complete'] else f' (stopped at {LIMIT})')
                  + f', {sum(len(v) for d in base["trans"].values() for v in d.values())} transitions: same over both modules')
    missing = sorted(set(pidvar) - used)
    print(f'transitions: {"ok" if ok else "FAIL"} ({n} initial states; {len(used)} of {len(pidvar)} rules exercised'
          + (f'; not exercised: {" ".join(missing)}' if missing else '')
          + f'; canStep agrees with the rules in {ncan} configurations)')
    return ok


def check_atomic(tmp):
    ok = True
    for rel, want, _, _ in ATOMIC:
        src = os.path.join(REPO, rel)
        d, text = os.path.dirname(src), open(src).read()
        if want is None:
            want = [x == 'Solution' for x in re.findall(r'^\*\*\* Expect: (No solution|Solution)\b', text, re.M)]
        got = [[s for s, _ in verdicts(maude(src, d)[0])]]
        for ks in (0, 2):
            p = os.path.join(tmp, f'atomic-{os.path.basename(rel)[:-len(".maude")]}-k{ks}.maude')
            with open(p, 'w') as f:
                f.write(gen.act_file(text, d, focus=[], ksize=ks))
            got.append([s for s, _ in verdicts(maude(p, tmp)[0])])
        good = all(g == want for g in got)
        ok &= good
        show = lambda v: ' '.join('yes' if x else 'no' for x in v)
        print(f'atomic: {rel}: {len(want)} searches, solution expected: {show(want)} | '
              + ('as expected over PROMELA-SEMANTICS and over PROMELA-SEMANTICS-ACT (propKsize 0, 2)' if good
                 else 'DIFFERENT: ' + ' / '.join(show(g) for g in got)))
    print(f'atomic: {"ok" if ok else "FAIL"}')
    return ok


def check_props(tmp):
    src = os.path.join(BP, 'dining-philosophers-analysis.maude')
    p = os.path.join(tmp, 'props.maude')
    with open(p, 'w') as f:
        f.write(gen.act_file(open(src).read(), BP))
        f.write('''
mod PROPS-CHECK is
  inc DINING-PROMELA-TEST .
  eq focusIds = mt .
  eq propKsize = 1 .
  var S : State .
  eq obs(S) = < obsUpto(S, plainProps ; execP(load-seq, *) ; execKP(*, body0)) > .
endm
red init |= execP(load-seq, *) .
red init |= execKP(*, body0) .
search [1] init =>1 S:GlobalState such that (S:GlobalState |= execP(load-seq, *)) and (S:GlobalState |= execKP(*, body0)) .
red obs(init) .
''')
    out, _ = maude(p, tmp)
    r = results(out)
    ok = (len(r) == 4 and r[0] == r[1] == 'resultBool:false' and r[2] == 'search: solution'
          and 'execP(load-seq,*):false' in r[3] and 'execKP(*,' in r[3])
    print(f'props: {"ok" if ok else "FAIL"} (false in init, true after [load-seq] of process 0, listed in obs(init))')
    if not ok:
        print('\n'.join(x[:400] for x in r))
    return ok


def main():
    write_hook = '--write-hook' in sys.argv
    if write_hook: sys.argv.remove('--write-hook')
    checks = dict(sync=lambda t: check_sync(write_hook), prefix=check_prefix, tests=check_tests, safety=check_safety,
                  transitions=check_transitions, atomic=check_atomic, props=check_props)
    names = sys.argv[1:] or list(checks)
    with tempfile.TemporaryDirectory() as tmp:
        ok = [checks[n](tmp) for n in names]
    sys.exit(0 if all(ok) else 1)


if __name__ == '__main__':
    main()
