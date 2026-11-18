import maude
from AGCEL.MaudeEnv import MaudeEnv
from AGCEL.QLearning import QLearner
from AGCEL.AStar import *
from AGCEL.DQNLearning import DQNLearner, READOUTS
from AGCEL.common import build_vocab, extract_predicate_vector, make_encoder, compare_qtable_dqn
import os, sys, re, json, time, subprocess, random
import torch
torch.set_grad_enabled(False)

# Usage:
# python3 test.py <maude_model> <init_term> <goal_prop> <qtable_file> [--dqn-readout per-obs|all]

def run_bfs(m, env, n0):
    """run BFS (search with score 0 for all states)"""
    V0 = lambda obs_term, g_state=None: 0
    V0.needs_obs = False

    print('\nMethod: BFS')
    print('-' * 40)
    start_time = time.perf_counter()
    res0 = Search().search(n0, V0, 9999)
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000
    print(f'  States          : {res0[2]}')
    print(f'  Time            : {elapsed_ms:.3f} ms')
    print(f'  Goal            : {"reached" if res0[0] else "not reached"}')

def run_random(m, env, n0, n_runs=10, base_seed=42):
    """run search with random score, averaged over n_runs with fixed seeds"""
    print('\nMethod: Random')
    print('-' * 40)

    total_states = 0
    total_time = 0
    goal_count = 0

    for i in range(n_runs):
        seed = base_seed + i
        rng = random.Random(seed)
        Vr = lambda obs_term, g_state=None, r=rng: r.random()
        Vr.needs_obs = False

        start_time = time.perf_counter()
        res = Search().search(n0, Vr, 9999)
        end_time = time.perf_counter()

        total_states += res[2]
        total_time += (end_time - start_time) * 1000
        if res[0]:
            goal_count += 1

    avg_states = total_states / n_runs
    avg_time = total_time / n_runs
    print(f'  Runs            : {n_runs} (seeds {base_seed}-{base_seed + n_runs - 1})')
    print(f'  States (avg)    : {avg_states:.1f}')
    print(f'  Time (avg)      : {avg_time:.3f} ms')
    print(f'  Goal            : {goal_count}/{n_runs} reached')

def assert_same_vocabulary(saved, env, source):
    """the propositions a value was learned over must be those of the model it is used on, as an
    ordered list: with other propositions the value would be read for the wrong observations"""
    here = build_vocab(env)
    if list(saved) != here:
        only_value = [x for x in saved if x not in here][:3]
        only_model = [x for x in here if x not in saved][:3]
        raise ValueError(f'{source}: the value was learned over {len(saved)} propositions and this model has {len(here)}; '
                         f'only in the value: {only_value}; only in the model: {only_model}')


def run_qtable(m, env, n0, qtable_file):
    """run search with qtable heuristic"""
    learner = QLearner()
    learner.load_value_function(qtable_file + '.agcel', m)
    for t in learner.v_dict:   # every stored observation must list the propositions of this model
        assert_same_vocabulary(sorted(name for name, _ in extract_predicate_vector(t)), env, qtable_file + '.agcel')
    V = learner.get_value_function()

    print('\nMethod: QTable')
    print('-' * 40)
    start_time = time.perf_counter()
    res = Search().search(n0, V, 9999)
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000
    print(f'  States          : {res[2]}')
    print(f'  Time            : {elapsed_ms:.3f} ms')
    print(f'  Goal            : {"reached" if res[0] else "not reached"}')

def run_dqn(m, env, n0, qtable_file):
    """load DQN model for search"""
    mobj = re.search(r'(.+?)(-c|-o\d+|-oracle)?$', qtable_file)
    base_prefix = mobj.group(1) + mobj.group(2) if mobj.group(2) else qtable_file

    dqn_model_file = base_prefix + '-d.pt'
    dqn_vocab_file = base_prefix + '-v.json'

    with open(dqn_vocab_file, 'r') as f:
        vocab = json.load(f)
    assert_same_vocabulary(vocab, env, dqn_vocab_file)

    dqn = DQNLearner(
        state_encoder=make_encoder(vocab),
        input_dim=len(vocab),
        num_actions=len(env.rules),
        gamma=0.95
    )

    dqn.load(dqn_model_file)
    dqn.q_network.eval()

    return dqn

def run_dqn_mode(m, env, n0, qtable_file, mode="dqn", readout="per-obs"):
    """
    run search with DQN heuristic
    mode: "zero" (all zero), "random" (random value), "dqn" (trained value)
    readout: how the trained value is read from the network outputs ("per-obs" or "all")
    """
    dqn = run_dqn(m, env, n0, qtable_file)
    dqn.value_cache.clear()
    V_dqn = dqn.get_value_function(mode=mode, readout=readout)

    mode_names = {"zero": "DQN-Zero", "random": "DQN-Random", "dqn": "DQN"}
    method_name = mode_names.get(mode, f"DQN-{mode.upper()}")

    print(f'\nMethod: {method_name}')
    print('-' * 40)
    if mode == "dqn":
        print(f'  Readout         : {readout}')
    start_time = time.perf_counter()
    res = Search().search(n0, V_dqn, 9999)
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000
    print(f'  States          : {res[2]}')
    print(f'  Time            : {elapsed_ms:.3f} ms')
    print(f'  Goal            : {"reached" if res[0] else "not reached"}')

    # compare qtable and DQN value order
    if mode == "dqn":
        compare_qtable_dqn(qtable_file, dqn, m, readout=readout)


if __name__ == "__main__":
    readout = "per-obs"
    if "--dqn-readout" in sys.argv:
        i = sys.argv.index("--dqn-readout")
        readout = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    if readout not in READOUTS:
        sys.exit(f'--dqn-readout must be one of {READOUTS}')

    model = sys.argv[1]
    init  = sys.argv[2]
    prop  = sys.argv[3]
    qtable_file = sys.argv[4]

    # fixed seed for reproducible runs (AGCEL_SEED); unset = nondeterministic
    _seed = os.environ.get("AGCEL_SEED")
    if _seed is not None:
        _seed = int(_seed)
        import numpy as _np
        random.seed(_seed); _np.random.seed(_seed); torch.manual_seed(_seed)

    mode = os.environ.get("MODE")
    if mode:
        maude.init()
        maude.load(model)
        m = maude.getCurrentModule()
        env = MaudeEnv(m, prop, lambda: init)
        init_term = m.parseTerm(init); init_term.reduce()
        n0 = Node(m, init_term)

        if mode == "bfs":
            print('\nTest')
            print('-' * 40)
            print(f'  Module          : {m}')
            print(f'  Init            : {init}')
            print(f'  Goal            : {prop}')
            print(f'  QTable          : {qtable_file}')
            if _seed is not None:
                print(f'  Seed            : {_seed}')

        if mode == "bfs":
            run_bfs(m, env, n0)
        elif mode == "random":
            run_random(m, env, n0)
        elif mode == "qtable":
            run_qtable(m, env, n0, qtable_file)
        elif mode in ("dqn", "dqn-zero", "dqn-random"):
            dqn_mode = mode.replace("dqn-", "") if "-" in mode else "dqn"
            run_dqn_mode(m, env, n0, qtable_file, mode=dqn_mode, readout=readout)
        sys.exit(0)

    for mode in ["bfs", "random", "qtable", "dqn"]:
        envp = os.environ.copy(); envp["MODE"] = mode
        p = subprocess.Popen(
            [sys.executable, sys.argv[0], model, init, prop, qtable_file, "--dqn-readout", readout],
            env=envp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        out, err = p.communicate()
        if out: print(out, end="")
        if err: print(err, file=sys.stderr, end="")