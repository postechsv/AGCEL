import maude
from AGCEL.MaudeEnv import MaudeEnv
from AGCEL.QLearning import QLearner
from AGCEL.AStar import *
from AGCEL.DQNLearning import DQNLearner
from AGCEL.common import make_encoder
import os, sys, re, json, time, subprocess
import torch
torch.set_grad_enabled(False)

def run_baseline(m, env, n0):
    V0 = lambda obs_term, g_state=None: 0
    V0.needs_obs = False

    print('\n=== SEARCH WITHOUT TRAINING ===')
    start_time = time.perf_counter()
    res0 = Search().search(n0, V0, 9999)
    end_time = time.perf_counter()
    print('[BASELINE] n_states:', res0[2])
    print(f'[BASELINE] Elapsed time: {(end_time - start_time)*1000:.3f} ms')
    if res0[0]: print('[BASELINE] Goal reached!')

def run_qtable(m, env, n0, qtable_file):
    learner = QLearner()
    learner.load_value_function(qtable_file + '.agcel', m)
    V = learner.get_value_function()

    print('\n=== SEARCH WITH QTABLE ===')
    start_time = time.perf_counter()
    res = Search().search(n0, V, 9999)
    end_time = time.perf_counter()
    print('[QTABLE] n_states:', res[2])
    print(f'[QTABLE] Elapsed time: {(end_time - start_time)*1000:.3f} ms')
    if res[0]: print('[QTABLE] Goal reached!')

def run_dqn(m, env, n0, qtable_file, extra_args):
    mobj = re.search(r'(.+?)(-c|-o\d+|-oracle)?$', qtable_file)
    base_prefix = mobj.group(1) if mobj else qtable_file

    if len(extra_args) == 6:
        lr = float(extra_args[0])
        gamma = float(extra_args[1])
        tau = float(extra_args[2])
        end = float(extra_args[3])
        decay = float(extra_args[4])
        tf = int(extra_args[5])

        suffix_parts = [
            f'lr={lr}', f'gamma={gamma}', f'tau={tau}',
            f'end={end}', f'decay={decay}', f'tf={tf}'
        ]
        suffix = '-' + '-'.join(suffix_parts)
        dqn_model_file = base_prefix + f'-d{suffix}.pt'
        dqn_vocab_file = base_prefix + f'-v{suffix}.json'
    else:
        dqn_model_file = base_prefix + '-d.pt'
        dqn_vocab_file = base_prefix + '-v.json'

    with open(dqn_vocab_file, 'r') as f:
        vocab = json.load(f)

    dqn = DQNLearner(
        state_encoder=make_encoder(vocab),
        input_dim=len(vocab),
        num_actions=len(env.rules),
        learning_rate=5e-4,
        gamma=0.99,
        tau=0.001,
        epsilon_end=0.05,
        epsilon_decay=0.0002,
        target_update_frequency=500
    )

    dqn.load(dqn_model_file)
    dqn.q_network.eval()
    V_dqn = dqn.get_value_function()

    print('\n=== SEARCH WITH DQN ===')
    start_time = time.perf_counter()
    res_dqn = Search().search(n0, V_dqn, 9999)
    end_time = time.perf_counter()
    print('[DQN] n_states:', res_dqn[2])
    print(f'[DQN] Elapsed time: {(end_time - start_time)*1000:.3f} ms')
    print('[DQN] Goal reached!' if res_dqn[0] else '[DQN] Goal not reached')

if __name__ == "__main__":
    model = sys.argv[1]
    init  = sys.argv[2]
    prop  = sys.argv[3]
    qtable_file = sys.argv[4]
    extra_args = sys.argv[5:]  # [lr gamma tau end decay tf]

    mode = os.environ.get("MODE")
    if mode:
        maude.init()
        maude.load(model)
        m = maude.getCurrentModule()
        env = MaudeEnv(m, prop, lambda: init)
        init_term = m.parseTerm(init); init_term.reduce()
        n0 = Node(m, init_term)

        print('\n=== TEST SETUP ===')
        print(f'Module: {m}')
        print(f'Init term: {init}')
        print(f'Goal proposition: {prop}')
        print(f'QTable file: {qtable_file}')
        if extra_args:
            print(f'Sweep parameters: {extra_args}')

        if mode == "baseline":
            run_baseline(m, env, n0)
        elif mode == "qtable":
            run_qtable(m, env, n0, qtable_file)
        elif mode == "dqn":
            run_dqn(m, env, n0, qtable_file, extra_args)
        sys.exit(0)

    for mode in ["baseline", "qtable", "dqn"]:
        envp = os.environ.copy(); envp["MODE"] = mode
        p = subprocess.Popen(
            [sys.executable, sys.argv[0], model, init, prop, qtable_file] + extra_args,
            env=envp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        out, err = p.communicate()
        if out: print(out, end="")
        if err: print(err, file=sys.stderr, end="")