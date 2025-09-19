import maude
from AGCEL.MaudeEnv import MaudeEnv
from AGCEL.QLearning import QLearner
from AGCEL.DQNLearning import DQNLearner
from AGCEL.common import build_vocab, make_encoder
import os, sys, json, time, subprocess

# Usage: python3 train.py <maude_model> <init_term> <goal_prop> <num_samples> <output_file_prefix> [trace_path]
# e.g., python3 train.py benchmarks/filter-analysis.maude init twoCrits 500 trained/filter-init3-twoCrits-500 traces/filter-init3-twoCrits-1.trace

def run_oracle():
    print('\n=== [WITH ORACLE] ===')
    learner = QLearner()
    t0 = time.time()
    learner.pretrain(env, trace_path)
    size_before = learner.get_size()
    learner.train(env, num_samples)
    t1 = time.time()
    suffix = "-o" + trace_path.split("-")[-1].split(".")[0] if "-" in trace_path else "-oracle"
    out_file = output_pref + suffix + '.agcel'
    learner.dump_value_function(out_file)
    print(f'[Warm] Training time: {t1 - t0:.2f}s, # Entries: {size_before} -> {learner.get_size()}')
    print(f'       Value function: {os.path.basename(out_file)}')

def run_cold():
    print('\n=== [WITHOUT ORACLE] ===')
    learner = QLearner()
    t2 = time.time()
    learner.train(env, num_samples)
    t3 = time.time()
    out_file = output_pref + "-c.agcel"
    learner.dump_value_function(out_file)
    print(f'[Cold] Training time: {t3 - t2:.2f}s, # Entries: {learner.get_size()}')
    print(f'       Value function: {os.path.basename(out_file)}')

def run_dqn():
    print('\n=== [DQN] ===')
    vocab = build_vocab(env)
    dqn = DQNLearner(env, state_encoder=make_encoder(vocab),
                     input_dim=len(vocab), num_actions=len(env.rules),
                     gamma=0.95, lr=1e-3, tau=0.01)
    t4 = time.time()
    dqn.train(n_training_episodes=num_samples)
    t5 = time.time()

    model_file = output_pref + '-dqn.pt'
    vocab_file = output_pref + '-dqn-vocab.json'
    dqn.save_model(model_file)
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f)

    print(f'[DQN]  Training time: {t5 - t4:.2f}s')
    print(f'       Model: {os.path.basename(model_file)}')
    print(f'       Vocab: {os.path.basename(vocab_file)}')

if __name__ == "__main__":
    model_path   = sys.argv[1]
    init_term    = sys.argv[2]
    goal_prop    = sys.argv[3]
    num_samples  = int(sys.argv[4])
    output_pref  = sys.argv[5]
    trace_path   = sys.argv[6] if len(sys.argv) >= 7 else None

    mode = os.environ.get("MODE")
    if mode:
        maude.init()
        maude.load(model_path)
        m = maude.getCurrentModule()
        env = MaudeEnv(m, goal_prop, lambda: init_term)

        print('\n=== TRAINING SETUP ===')
        print(f'Module: {m}')
        print(f'Init term: {init_term}')
        print(f'Goal proposition: {goal_prop}')
        print(f'Training samples: {num_samples}')
        print(f'Trace file: {trace_path}')
        print(f'Output prefix: {output_pref}')

        if mode == "oracle":
            if trace_path is None:
                print("[WITH ORACLE] skipped (no trace provided)")
            else:
                run_oracle()
        elif mode == "cold":
            run_cold()
        elif mode == "dqn":
            run_dqn()
        sys.exit(0)

    modes = []
    if trace_path is not None:
        modes.append("oracle")
    modes += ["cold", "dqn"]

    for mname in modes:
        envp = os.environ.copy(); envp["MODE"] = mname
        p = subprocess.run(
            [sys.executable, sys.argv[0], model_path, init_term, goal_prop, str(num_samples), output_pref] + ([trace_path] if trace_path else []),
            env=envp, capture_output=True, text=True
        )
        if p.stdout: print(p.stdout, end="")
        if p.stderr: print(p.stderr, file=sys.stderr, end="")