import maude
from AGCEL.MaudeEnv import MaudeEnv
from AGCEL.QLearning import QLearner
from AGCEL.AStar import *
from AGCEL.DQNLearning import DQNLearner
from AGCEL.common import make_encoder, compare_qtable_dqn
import os, sys, re, json, time, subprocess, random
import torch
torch.set_grad_enabled(False)

# Usage:
# python3 test.py <maude_model> <init_term> <goal_prop> <qtable_file>

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

def run_qtable(m, env, n0, qtable_file):
    """run search with qtable heuristic"""
    learner = QLearner()
    learner.load_value_function(qtable_file + '.agcel', m)
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

    dqn = DQNLearner(
        state_encoder=make_encoder(vocab),
        input_dim=len(vocab),
        num_actions=len(env.rules),
        gamma=0.95
    )

    dqn.load(dqn_model_file)
    dqn.q_network.eval()

    return dqn

def run_dqn_mode(m, env, n0, qtable_file, mode="dqn"):
    """
    run search with DQN heuristic
    mode: "zero" (all zero), "random" (random value), "dqn" (trained value)
    """
    dqn = run_dqn(m, env, n0, qtable_file)
    dqn.value_cache.clear()
    V_dqn = dqn.get_value_function(mode=mode)

    mode_names = {"zero": "DQN-Zero", "random": "DQN-Random", "dqn": "DQN"}
    method_name = mode_names.get(mode, f"DQN-{mode.upper()}")

    print(f'\nMethod: {method_name}')
    print('-' * 40)
    start_time = time.perf_counter()
    res = Search().search(n0, V_dqn, 9999)
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000
    print(f'  States          : {res[2]}')
    print(f'  Time            : {elapsed_ms:.3f} ms')
    print(f'  Goal            : {"reached" if res[0] else "not reached"}')

    # compare qtable and DQN value order
    if mode == "dqn":
        compare_qtable_dqn(qtable_file, dqn, m)


if __name__ == "__main__":
    model = sys.argv[1]
    init  = sys.argv[2]
    prop  = sys.argv[3]
    qtable_file = sys.argv[4]

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

        if mode == "bfs":
            run_bfs(m, env, n0)
        elif mode == "random":
            run_random(m, env, n0)
        elif mode == "qtable":
            run_qtable(m, env, n0, qtable_file)
        elif mode in ("dqn", "dqn-zero", "dqn-random"):
            dqn_mode = mode.replace("dqn-", "") if "-" in mode else "dqn"
            run_dqn_mode(m, env, n0, qtable_file, mode=dqn_mode)
        sys.exit(0)

    for mode in ["bfs", "random", "qtable", "dqn"]:
        envp = os.environ.copy(); envp["MODE"] = mode
        p = subprocess.Popen(
            [sys.executable, sys.argv[0], model, init, prop, qtable_file],
            env=envp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        out, err = p.communicate()
        if out: print(out, end="")
        if err: print(err, file=sys.stderr, end="")