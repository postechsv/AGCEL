#!/usr/bin/env python3
"""Proposition lists of an arm, generated from the program text of a model and from the rules.

The lists say what a process is about to do (kPre: the first propKsize items of its
computation) or what was done last (execP: the rule label; execKP: the first propKsize items
of the computation the acting process had). Which items can occur follows from the text of
each process alone. For every process of the initial configuration the generator closes its
computation, taken by itself, under what each rule of the semantics does to the computation
of the process it acts on:

  [load-seq] [load-atomic] [load-if] [load-do] [load-basic] [load-goto] [load-break]
        a statement at the head becomes load(B, ...), for B = true and for B = false
  [fire]      a loaded branch @[ BS ~> K ] becomes effect(BS, nextPid) ~> K, for every branch
  [hs-fire]   (models with a rendezvous channel) the sender's branch becomes @[ # true ~> K ],
              the receiver's #toss(pid) ~> #recvMsg(...) ~> K or #recvMsg(...) ~> K
  [rel] [toss1] [toss2] [break] [break-prop] and the assign, send, recv and recvMsg rules
        on #rel, #toss, break, #assign, #send, #recv, #recvMsg at the head

Guards are not evaluated: every branch is followed, so an item the model never reaches may be
listed. What is known only at run time -- the value an assignment stores, the channel a name
stands for, the message received, the pid a lock is tossed back to -- is a variable of the
item, and a proposition holds when a computation matches its item. No state of the model is
built and no rule is applied; split, load, effect and prefix are reduced in Maude. Rules of
process creation and declarations (#run, #init, goto with a goto map, ...) are not covered:
the generator stops when a computation starts with one of them.

Arms and lists. execL: execP(l, *) for every rule label l. execK<n>: that and execAt(i) for
the items of the acting process. kpre<n>, kitem<n>: kAt(i) for the items of every process, as
the processes are written (names of single processes occur in the items).

Role lists (arms krole<n>, kroleq<n>, khead<n>, klabel<n>). Processes whose texts differ only
in names and numbers are instances of one template: the positions where they differ are the
roles, numbered in order of first occurrence. The closure of the template gives positions,
and rSome(j) says that some process is at position j of its own instance of the template. It
mentions no process and no name of a single process, so the list is the same for every number
of processes as long as the template is. kroleq<n> adds rTwo(j), at least two processes, and
rAll(j), all of them. khead<n>: positions that differ only in what follows the head statement
of a loaded branch are one position. klabel<n>: a position is a rule label, and a process is at
it when that rule acts on its computation.

The propositions (kAt, execAt, rSome, rTwo, rAll) are defined in semantics/promela-agcel.maude
over kItem, xItem and rItem; the module ARM-PROPS written here gives those three their
equations, one per item, and the lists.

  gen_promela_props.py gen <tag> <bench> <arm>      write models/<bench>_<arm>.props.maude (+ .json)
  gen_promela_props.py sizes [BENCH ...] [--arms ARM ...]     inputs and list sizes per model
  gen_promela_props.py check [BENCH ...] [--arms ARM ...] [--cap N]
        coverage of the generated lists by what an enumeration of the model finds (the
        reference: collect_props() of the driver; for role arms an enumeration that keeps the
        pid). The enumeration is never used to produce a list. For role arms also whether the
        vocabulary of a family is the same for every size given. Exit 1 on a failure.
"""
import argparse, hashlib, json, os, re, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LOAD_RULE = {'_;_': 'load-seq', 'atomic`{_`}': 'load-atomic', 'if_fi': 'load-if', 'do_od': 'load-do',
             'goto_': 'load-goto', 'break': 'load-break'}
NOT_COVERED = ('#wait', '#done', '#run', '#init', '#repeat', '#proctype', '#activeProctype', '#activeN', '#globals',
               '#freeze', '#sendMsg')
BRANCH, BRANCHES, NO_BRANCH = '@`[_`]', '__', '.BranchSet'
# variables that stand for run-time values in an item
V_INDEX, V_VALUE, V_CHAN, V_MSG, V_RECV, V_PID, V_REST = ('IV:Value', 'AV:Value', 'CV:Value', 'MV:ValueList', 'RV:Value',
                                                         'TI:Int', 'KC:K')
ROLE_ID, ROLE_NUM = "'ROLE", 900000
TOKEN = re.compile(r"('[^\s,()\[\]{}`]+)|(s_\^\d+\(0\)|s_\(0\)|(?<![\w^'.:])0(?![\w(]))")

AUX = '''
mod PROPS-GEN-AUX is
  inc {module} .
  op isVal : Expr -> Bool .
  eq isVal(V:Value) = true .
  eq isVal(E:Expr) = false [owise] .
  op isBasic : K -> Bool .
 ceq isBasic(BS:BasicStmt) = true if not (BS:BasicStmt :: Value) /\\ not (BS:BasicStmt :: Decl) .
  eq isBasic(K:K) = false [owise] .
endm
'''


class Unsupported(Exception):
    pass


def join(items):
    """the computation with these items, in prefix form"""
    if not items: return '.K'
    s = items[-1]
    for x in reversed(items[:-1]):
        s = f'_~>_({x}, {s})'
    return s


def comma(items):
    if not items: return 'nil'
    s = items[-1]
    for x in reversed(items[:-1]):
        s = f'_`,_({x}, {s})'
    return s


class Closure:
    """the computations one process can have, from the rules as they act on a computation"""

    def __init__(self, m, next_pid, rendezvous):
        self.m, self.next_pid, self.hs = m, next_pid, rendezvous
        self.cache = {}

    def term(self, s):
        """the term a string stands for, reduced: only then are the arguments of an associative
        operator (the items of a computation, the branches of a branch set) all at one level"""
        t = self.m.parseTerm(s)
        if t is None:
            raise ValueError('cannot parse: ' + s[:300])
        t.reduce()
        return t

    red = term

    def items(self, t):
        s = str(t.symbol())
        if s == '.K': return ()
        if s == '_~>_': return tuple(x.prettyPrint(0) for x in t.arguments())
        return (t.prettyPrint(0),)

    def is_true(self, s):
        return self.red(s).prettyPrint(0) == 'true'

    def args(self, s):
        t = self.term(s)
        return [x.prettyPrint(0) for x in t.arguments()]

    def steps(self, k):
        """[(rule label, computation after the rule)] for the computation k (a tuple of items)"""
        if not k: return []
        head, rest = k[0], list(k[1:])
        sym = str(self.term(head).symbol())
        out = []
        after = lambda s: self.items(self.red(s))
        if sym in NOT_COVERED:
            raise Unsupported(f'a computation starts with {sym}: process creation and declarations are not covered')
        if sym in LOAD_RULE or self.is_true(f'isBasic({head})'):
            first = f'split({head})' if sym == '_;_' else head
            for b in ('true', 'false'):
                out.append((LOAD_RULE.get(sym, 'load-basic'), after(f'load({b}, {join([first] + rest)})')))
            if sym == 'break' and rest:
                nxt = str(self.term(rest[0]).symbol())
                out.append(('break', tuple(rest[1:])) if nxt == 'do_od' else ('break-prop', (head, *rest[1:])))
            if sym == 'goto_':
                raise Unsupported('goto: the goto map of a proctype is not covered')
        elif sym in (BRANCH, BRANCHES):
            for b in ([head] if sym == BRANCH else self.args(head)):
                bt = self.term(b)
                inner = self.items(list(bt.arguments())[0])
                bs, kl = inner[0], list(inner[1:])
                out.append(('fire', after(join([f'effect({bs}, {self.next_pid})'] + kl + rest))))
                bsym = str(self.term(bs).symbol())
                if self.hs and bsym == '_!_':
                    out.append(('hs-fire', (f'{BRANCH}({join(["#_(true)"] + kl)})', *rest)))
                if self.hs and bsym == '_?_':
                    el = self.args(bs)[1]
                    msg = f'#recvMsg({el}, m({V_MSG}))'
                    out.append(('hs-fire', (f'#toss({V_PID})', msg, *kl)))
                    out.append(('hs-fire', (msg, *kl)))
        elif sym == '#rel':
            out.append(('rel', tuple(rest)))
        elif sym == '#toss':
            if rest and str(self.term(rest[0]).symbol()) == '#rel':
                out.append(('toss2', tuple(rest[1:])))
            elif rest:
                out.append(('toss1', (rest[0], head, *rest[1:])))
        elif sym == '#assign':
            x, i, v = self.args(head)
            if not self.is_true(f'isVal({i})'):
                out.append(('assign-eval-index', (f'#assign({x}, {V_INDEX}, {v})', *rest)))
            elif not self.is_true(f'isVal({v})'):
                out.append(('assign-eval-value', (f'#assign({x}, {i}, {V_VALUE})', *rest)))
            else:
                out += [('assign-store', tuple(rest)), ('assign-chan', tuple(rest))]
        elif sym == '#send':
            c, el = self.args(head)
            if not self.is_true(f'isVal({c})'):
                out.append(('send-eval-chan', (f'#send({V_CHAN}, {el})', *rest)))
            else:
                out.append(('send-push', tuple(rest)))
        elif sym == '#recv':
            c, el = self.args(head)
            if not self.is_true(f'isVal({c})'):
                out.append(('recv-eval-chan', (f'#recv({V_CHAN}, {el})', *rest)))
            else:
                out.append(('recv-pop', (f'#recvMsg({el}, m({V_MSG}))', *rest)))
        elif sym == '#recvMsg':
            el = self.term(self.args(head)[0])
            es = [] if str(el.symbol()) == 'nil' else ([x.prettyPrint(0) for x in el.arguments()]
                                                         if str(el.symbol()) == '_`,_' else [el.prettyPrint(0)])
            if not es:
                out.append(('recvMsg-done', tuple(rest)))
            else:
                e, more = es[0], es[1:]
                nxt = f'#recvMsg({comma(more)}, m({V_MSG}))' if more else '#recvMsg(nil, m(nil))'
                esym = str(self.term(e).symbol())
                if self.is_true(f'isVal({e})'):
                    out.append(('recvMsg-match', (nxt, *rest)))
                elif esym == '_`[_`]':
                    x, i = self.args(e)
                    out.append(('recvMsg-assign', (f'#assign({x}, {i}, {V_RECV})', nxt, *rest)))
                else:
                    out.append(('recvMsg-assign-simple', (f'#assign({e}, #_(0), {V_RECV})', nxt, *rest)))
        return out

    def close(self, k0):
        """{computation: labels of the rules that act on it}, for everything reachable from k0"""
        k0 = tuple(k0)
        if k0 in self.cache: return self.cache[k0]
        seen, todo = {}, [k0]
        while todo:
            k = todo.pop()
            if k in seen: continue
            st = self.steps(k)
            seen[k] = sorted({l for l, _ in st})
            todo += [n for _, n in st if n not in seen]
        self.cache[k0] = seen
        return seen

    def prefix(self, k, n):
        """(prefix form, readable form) of the first n items of k"""
        t = self.red(f'prefix({join(list(k))}, {n})')
        return t.prettyPrint(0), ' '.join(str(t).split())


def processes(m):
    """(pid -> items of its computation, nextPid, whether a name stands for a rendezvous channel)
    of the initial configuration"""
    t = m.parseTerm('init'); t.reduce()
    procs, npid, hs = {}, None, False
    for cell in list(t.arguments())[0].arguments():
        c = str(cell.symbol())
        if c == 'nextPid':
            npid = list(cell.arguments())[0].prettyPrint(0)
        elif c == 'procs':
            ps = list(cell.arguments())[0]
            for pr in ([ps] if str(ps.symbol()) == 'proc' else list(ps.arguments())):
                d = {str(x.symbol()): list(x.arguments())[0] for x in list(pr.arguments())[0].arguments()}
                procs[int(str(d['pid']))] = d['k']
                hs |= 'hs(' in d['env'].prettyPrint(0)
    return dict(sorted(procs.items())), npid, hs


def tokens(s):
    """[(kind, text)]: kind 'q' a quoted identifier, 'n' a numeral, '' the text between them"""
    out, pos = [], 0
    for mt in TOKEN.finditer(s):
        out.append(('', s[pos:mt.start()]))
        out.append(('q' if mt.group(1) else 'n', mt.group(0)))
        pos = mt.end()
    out.append(('', s[pos:]))
    return out


def numeral(n):
    return '0' if n == 0 else 's_(0)' if n == 1 else f's_^{n}(0)'


def templates(texts):
    """the processes as instances of templates. texts: pid -> text of its computation (prefix
    form). Returns [(template text, {pid: {role token: token of that process}})]: processes whose
    texts differ only in identifiers and numerals share a template, in which the positions
    where they differ hold role tokens numbered in order of first occurrence."""
    shape = {}
    for pid, s in texts.items():
        tk = tokens(s)
        assert ROLE_ID not in s and not any(k == 'n' and len(x) > 9 for k, x in tk), 'a name or number of the model looks like a role'
        shape.setdefault(tuple((k, x if not k else None) for k, x in tk), []).append((pid, tk))
    out = {}
    for group in shape.values():
        n = len(group[0][1])
        varying = [i for i in range(n) if group[0][1][i][0] and len({tk[i][1] for _, tk in group}) > 1]
        for pid, tk in group:
            role, parts = {}, [x for _, x in tk]
            for i in varying:
                kind, x = tk[i]
                if (kind, x) not in role:
                    r = len(role) + 1
                    role[kind, x] = f'{ROLE_ID}{r}' if kind == 'q' else numeral(ROLE_NUM + r)
                parts[i] = role[kind, x]
            out.setdefault(''.join(parts), {})[pid] = {r: x for (_, x), r in role.items()}
    return sorted(out.items())


def instance(item, sub):
    """a template item with the role tokens replaced by the tokens of one process"""
    return ''.join(sub.get(x, x) if k else x for k, x in tokens(item))


def readable_roles(s):
    """role tokens of a readable (mixfix) item written as <1>, <2>, ..."""
    s = re.sub(r"'ROLE(\d+)", r'<\1>', s)
    return re.sub(r'\b9000(\d\d)\b', lambda mt: f'<{int(mt.group(1))}>', s)


def head_only(c, item):
    """the item with what follows the head statement of each loaded branch replaced by a variable"""
    sym = str(c.term(item).symbol())
    if sym not in (BRANCH, BRANCHES):
        return item
    bs = []
    for i, b in enumerate([item] if sym == BRANCH else c.args(item)):
        bt = c.term(b)
        inner = c.items(list(bt.arguments())[0])
        bs.append(f'{BRANCH}(_~>_({inner[0]}, KC{i + 1}:K))')
    s = bs[-1]
    for x in reversed(bs[:-1]):
        s = f'__({x}, {s})'
    return s


def item_block(fn, prop, items, about):
    L = [''] + [f'  --- {a}' for a in about]
    for i, (item, show) in enumerate(items, 1):
        L += [f'  --- {prop}({i}): {show}', f'  eq {fn}({i}, {item}) = true .']
    return L


def plist(name, prop, n):
    return [f'  op {name} : -> PropList .',
            f'  eq {name} = ' + (' ; '.join(f'{prop}({i})' for i in range(1, n + 1)) or 'nil') + ' .']


def generate(m, kind, ksize):
    """the lists of an arm. kind: execL | execK | kpre | kitem (the items of each process as it
    is written) or krole | kroleq | khead | klabel (positions of the template). Returns (text of
    the module ARM-PROPS, info). The propositions kAt, execAt, rSome, rTwo, rAll and the
    functions kItem, xItem, rItem they read are declared in semantics/promela-agcel.maude."""
    import maude
    with tempfile.NamedTemporaryFile('w', suffix='.maude', delete=False) as f:
        f.write(AUX.format(module=str(m)))
    maude.load(f.name); os.unlink(f.name)
    ma = maude.getModule('PROPS-GEN-AUX')
    procs, npid, hs = processes(ma)
    c = Closure(ma, npid, hs)
    L = ['mod ARM-PROPS is', '  pr MODEL .']
    n = ksize or 0
    texts = {pid: join(list(c.items(kt))) for pid, kt in procs.items()}
    info = dict(provenance='generated', generator='scripts/gen_promela_props.py', kind=kind, ksize=ksize,
                processes=len(procs), rendezvous=hs, next_pid=npid,
                input_sha256=hashlib.sha256(json.dumps(texts, sort_keys=True).encode()).hexdigest(),
                statements={pid: len(c.items(c.red(f'split({t})'))) if str(c.term(t).symbol()) == '_;_' else 1
                            for pid, t in texts.items()})

    if kind in ('execL', 'execK', 'kpre', 'kitem'):
        comps = {}
        for pid, kt in procs.items():
            for k, labels in c.close(c.items(kt)).items():
                comps.setdefault(k, set()).update(labels)
        info['computations'] = len(comps)
        if kind in ('kpre', 'kitem'):
            items = dict(c.prefix(k, n) for k in comps)
            L += item_block('kItem', 'kAt', sorted(items.items()),
                            ['kAt(i): the computation of some process begins with item i (its first propKsize items).',
                             'A variable in an item stands for a value known only at run time.'])
            L += plist('kPreProps', 'kAt', len(items))
            info['sizes'] = dict(kPreProps=len(items))
        else:
            labels = sorted(r.getLabel() for r in m.getRules() if r.getLabel())
            L += ['', '  --- execP(l, *) for every rule label l of the semantics',
                  '  op execLProps : -> PropList .', '  eq execLProps = ' + ' ; '.join(f'execP({l}, *)' for l in labels) + ' .']
            info.update(sizes=dict(execLProps=len(labels)), labels=labels)
            if kind == 'execK':
                items = dict(c.prefix(k, n) for k, ls in comps.items() if ls)
                items.setdefault('.K', '.K')   # [loose] records an empty computation
                L += item_block('xItem', 'execAt', sorted(items.items()),
                                ['execAt(i): the process that acted last had a computation beginning with item i (its first',
                                 'propKsize items) before the step.'])
                L += plist('execKProps', 'execAt', len(items))
                info['sizes']['execKProps'] = len(items)
    else:
        tpl = templates(texts)
        pos, desc = {}, {}     # position -> {pid: items of that process}, description
        for text, subs in tpl:
            for k, labels in c.close(c.items(c.term(text))).items():
                item, show = c.prefix(k, n)
                if kind == 'klabel':
                    keys = [(l, f'about to apply [{l}]') for l in labels] or [('~', 'no rule acts on the computation')]
                elif kind == 'khead':
                    h = head_only(c, item)
                    keys = [(h, readable_roles(' '.join(str(c.term(h)).split())))]
                else:
                    keys = [(item, readable_roles(show))]
                for key, d in keys:
                    desc[key] = d
                    for pid, sub in subs.items():
                        one = head_only(c, item) if kind == 'khead' else item
                        pos.setdefault(key, {}).setdefault(pid, set()).add(instance(one, sub))
        order = sorted(pos)
        vocab = [f'rSome({j}): {desc[it]}' for j, it in enumerate(order, 1)]
        L += ['', '  --- positions: rItem(j, pid, K) says that K is position j as process pid has it']
        for j, it in enumerate(order, 1):
            L.append(f'  --- {vocab[j - 1]}')
            for pid, insts in sorted(pos[it].items()):
                L += [f'  eq rItem({j}, {pid}, {inst}) = true .' for inst in sorted(insts)]
        L += plist('kRoleProps', 'rSome', len(order))
        info['sizes'] = dict(kRoleProps=len(order))
        if kind == 'kroleq':
            L += plist('kRole2Props', 'rTwo', len(order)) + plist('kRoleAllProps', 'rAll', len(order))
            info['sizes'].update(kRole2Props=len(order), kRoleAllProps=len(order))
        info.update(templates=len(tpl), processes_per_template=[len(sb) for _, sb in tpl], vocabulary=vocab,
                    vocabulary_sha256=hashlib.sha256('\n'.join(vocab).encode()).hexdigest())
    return '\n'.join(L + ['endm', '']), info


KIND = re.compile(r'(execL)|(execK|kpre|kitem|kroleq|krole|khead|klabel)(\d+)')


def arm_kind(arm):
    """(kind, propKsize) of an arm with generated lists"""
    mt = KIND.fullmatch(arm)
    if not mt: raise ValueError(f'no generated lists for arm {arm!r}')
    return ('execL', 0) if mt.group(1) else (mt.group(2), int(mt.group(3)))


def write(out, bench, arm):
    """generate the lists of an arm into out/models/<bench>_<arm>.props.maude (+ .json)"""
    import maude
    import run_extended_exp as drv
    maude.init(advise=False)
    maude.load(drv.wrapper(out, bench, arm, collect=True))
    m = maude.getCurrentModule()
    kind, ksize = arm_kind(arm)
    body, info = generate(m, kind, ksize)
    info.update(bench=bench, arm=arm, graph=drv.graph_of(arm))
    head = ['*** GENERATED by scripts/gen_promela_props.py from the program text -- do not edit by hand.',
            f'*** {bench}, arm {arm}: {info["graph"]}, propKsize {ksize}; sizes {info["sizes"]}', '']
    path = drv.props_file(out, bench, arm)
    drv.write_if_changed(path, '\n'.join(head) + body)
    drv.write_if_changed(path + '.json', json.dumps(info))
    return info


CHECK_AUX = '''
mod PROPS-CHECK is
  inc {module} .
  sort PidItem PidItems .  subsort PidItem < PidItems .
  op noItems : -> PidItems [ctor] .
  op __ : PidItems PidItems -> PidItems [ctor assoc comm id: noItems] .
  op at : Int K -> PidItem [ctor] .
  op pidItems : GlobalState -> PidItems .
  op pidAux : KConfig -> PidItems .
  var I : Int . var K : K . vars INNER PR CFG : KConfig .
  eq pidItems([ procs(PR) CFG ]) = pidAux(PR) .
  eq pidAux(proc(pid(I) INNER k(K)) PR) = at(I, prefix(K, propKsize)) pidAux(PR) .
  eq pidAux(PR) = noItems [owise] .
endm
'''


def check(benches, arms, cap):
    "coverage of the generated lists by the reference enumeration; role vocabularies across sizes"
    import subprocess
    ok, vocab = True, {}
    for b in benches:
        for arm in arms:
            r = subprocess.run([sys.executable, __file__, 'check1', b, arm, '--cap', str(cap)], capture_output=True, text=True,
                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            line = [l for l in r.stdout.splitlines() if l.startswith('{')]
            if r.returncode or not line:
                ok = False
                print(f'{b} {arm}: FAILED to run: {(r.stdout + r.stderr)[-800:]}', flush=True); continue
            x = json.loads(line[-1])
            good = x['coverage'] == 1.0
            ok &= good
            print(f"{b} {arm}: sizes {x['sizes']}; reference ({x['reference']}, {x['states']} states"
                  + ('' if x['complete'] else ', NOT exhaustive: stopped at the cap') + f") finds {x['enumerated']}, "
                  f"covered {x['covered']}: coverage {x['coverage']:.3f}" + ('' if good else f" MISSING {x['missing'][:3]}")
                  + f"; generated but never found: {x['unused']}", flush=True)
            if 'vocabulary_sha256' in x:
                fam = b.partition('.')[0].rpartition('-')[0] or b
                vocab.setdefault((fam, arm), []).append((b, x['vocabulary_sha256'], x['sizes']['kRoleProps']))
    for (fam, arm), vs in sorted(vocab.items()):
        if len(vs) > 1:
            same = len({h for _, h, _ in vs}) == 1
            print(f"{fam} {arm}: vocabulary over {', '.join(b for b, _, _ in vs)}: "
                  + ('byte-identical' if same else 'DIFFERENT (' + ', '.join(f'{b}: {n} positions' for b, _, n in vs) + ')'))
    return ok


def check1(bench, arm, cap):
    "one model and arm: the generated lists against what an enumeration of the model finds"
    import subprocess
    import maude
    import run_extended_exp as drv
    kind, ksize = arm_kind(arm)
    maude.init(advise=False)
    with tempfile.TemporaryDirectory() as work:
        for d in ('models', 'wrappers'): os.makedirs(os.path.join(work, 'gen', d))
        out = os.path.join(work, 'gen')
        info = write(out, bench, arm)
        maude.load(drv.wrapper(out, bench, arm))       # the module a job runs, with the generated lists
        run = maude.getCurrentModule()
        size, used, miss, want = info['sizes'], set(), [], 0
        red = lambda m, s: (lambda t: (t.reduce(), t.prettyPrint(0))[1])(m.parseTerm(s))

        def judge(hit, show):
            nonlocal want
            want += 1
            used.update(hit)
            if not hit: miss.append(' '.join(show.split())[:240])
        if kind in ('execL', 'execK', 'kpre', 'kitem'):
            # the reference is collect_props() of the driver, run in a process of its own
            r = subprocess.run([sys.executable, os.path.join(HERE, 'run_extended_exp.py'), 'props', 'ref', bench, arm,
                                '--cap', str(cap), '--walks', '0'], capture_output=True, text=True,
                               env=dict(os.environ, AGCEL_EXT_OUT=work, PYTHONDONTWRITEBYTECODE='1'))
            line = [l for l in r.stdout.splitlines() if l.startswith('{')]
            if r.returncode or not line:
                sys.exit('collect_props failed: ' + (r.stdout + r.stderr)[-800:])
            ref = json.loads(line[-1])
            reference, states, complete = 'collect_props', ref['states'], ref['complete']
            for prop, show in sorted(ref['items'].items()):
                t = run.parseTerm(prop)
                op, a = str(t.symbol()), [x.prettyPrint(0) for x in t.arguments()]
                if op == 'execP':
                    if kind in ('execL', 'execK'): judge([a[0]] if a[0] in info['labels'] else [], show)
                elif op == 'execKP' and kind == 'execK':
                    judge([i for i in range(1, size['execKProps'] + 1) if red(run, f'xItem({i}, {a[1]})') == 'true'], show)
                elif op == 'kPre' and kind in ('kpre', 'kitem'):
                    judge([i for i in range(1, size['kPreProps'] + 1) if red(run, f'kItem({i}, {a[1]})') == 'true'], show)
            first = 'execKProps' if kind == 'execK' else next(iter(size))
            if kind == 'execK':   # labels and items are two lists: count the unused of both
                used_n = len([u for u in used if isinstance(u, int)])
                unused = (size['execKProps'] - used_n) + (size['execLProps'] - len([u for u in used if not isinstance(u, int)]))
            else:
                unused = size[first] - len(used)
        else:
            # role positions need the pid of each computation: enumerate here
            aux = os.path.join(work, 'check.maude')
            with open(aux, 'w') as f:
                f.write(CHECK_AUX.format(module=str(run)))
            maude.load(aux)
            m = maude.getModule('PROPS-CHECK')
            found = set()

            def note(i, key, t):
                r = m.parseTerm(f'pidItems({key})'); r.reduce()
                if str(r.symbol()) == 'at':
                    found.add(r.prettyPrint(0))
                elif str(r.symbol()) == '__':
                    found.update(x.prettyPrint(0) for x in r.arguments())
            st = drv.explore(m, cap=cap, on_state=note)
            reference, states, complete = 'enumeration with pids', st['states'], st['complete']
            for x in sorted(found):
                t = m.parseTerm(x)
                pid, item = [a.prettyPrint(0) for a in t.arguments()]
                judge([i for i in range(1, size['kRoleProps'] + 1) if red(m, f'rItem({i}, {pid}, {item})') == 'true'],
                      f'process {pid}: ' + str(list(t.arguments())[1]))
            unused = size['kRoleProps'] - len(used)
        out_ = dict(bench=bench, arm=arm, sizes=size, enumerated=want, covered=want - len(miss),
                    coverage=(want - len(miss)) / want if want else 1.0, missing=miss, unused=unused,
                    reference=reference, states=states, complete=complete)
        if 'vocabulary_sha256' in info:
            out_['vocabulary_sha256'] = info['vocabulary_sha256']
        print(json.dumps(out_))


def sizes(benches, arms):
    "the inputs and the list sizes of the generator, per model (no enumeration)"
    import subprocess
    for b in benches:
        row = []
        for arm in arms:
            with tempfile.TemporaryDirectory() as work:
                r = subprocess.run([sys.executable, __file__, 'gen', 'sizes', b, arm], capture_output=True, text=True,
                                   env=dict(os.environ, AGCEL_EXT_OUT=work, PYTHONDONTWRITEBYTECODE='1'))
            line = [l for l in r.stdout.splitlines() if l.startswith('{')]
            if r.returncode or not line:
                row.append(f'{arm}: FAILED ({(r.stdout + r.stderr).strip().splitlines()[-1][:200]})'); continue
            x = json.loads(line[-1])
            if not row:
                row.append(f"{x['processes']} processes, statements per process {sorted(set(x['statements'].values()))}, "
                           f"rendezvous {x['rendezvous']}")
            row.append(f"{arm}: " + ', '.join(f'{k} {v}' for k, v in x['sizes'].items()))
        print(f'{b}: ' + ' | '.join(row), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    g = sp.add_parser('gen'); g.add_argument('tag'); g.add_argument('bench'); g.add_argument('arm')
    default_arms = ['execL', 'execK1', 'execK2', 'kpre1', 'kpre2', 'krole1', 'khead1', 'klabel1']
    default_benches = ['dining', 'filter', 'qlock', 'onethirdrule', 'chan', 'bakery', 'dining-3', 'filter-3', 'qlock-3',
                       'onethirdrule-3', 'dining-4', 'qlock-4']
    for name in ('check', 'sizes'):
        c = sp.add_parser(name); c.add_argument('benches', nargs='*', default=default_benches)
        c.add_argument('--arms', nargs='+', default=default_arms)
        c.add_argument('--cap', type=int, default=300000)
    o = sp.add_parser('check1'); o.add_argument('bench'); o.add_argument('arm'); o.add_argument('--cap', type=int, default=300000)
    a = ap.parse_args()
    if a.cmd == 'gen':
        import run_extended_exp as drv
        out = os.path.join(drv.OUT, a.tag)
        for d in ('models', 'wrappers'): os.makedirs(os.path.join(out, d), exist_ok=True)
        print(json.dumps(write(out, a.bench, a.arm)))
    elif a.cmd == 'check1':
        check1(a.bench, a.arm, a.cap)
    elif a.cmd == 'sizes':
        sizes(a.benches, a.arms)
    else:
        sys.exit(0 if check(a.benches, a.arms, a.cap) else 1)


if __name__ == '__main__':
    main()
