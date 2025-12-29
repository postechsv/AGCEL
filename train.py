import maude
from AGCEL.MaudeEnv import MaudeEnv
from AGCEL.QLearning import QLearner
from AGCEL.DQNLearning import DQNLearner
from AGCEL.common import build_vocab, make_encoder, compare_qtable_dqn
import os, sys, json, time, subprocess, numpy as np

# Usage:
# python3 train.py <maude_model> <init_term> <goal_prop> <num_samples> <output_file_prefix> [trace_path]

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

def run_dqn(learning_rate=5e-4,
            gamma=0.95,
            tau=0.01,
            epsilon_end=0.05,
            epsilon_decay=0.0005,
            target_update_frequency=50,
            goal_ratio=0.2,
            sweep_suffix=None):
    print('\n=== [DQN] ===')
    if sweep_suffix is not None:
        print(f'Hyperparameters: lr={learning_rate}, gamma={gamma}, tau={tau}, '
            f'eps_end={epsilon_end}, eps_decay={epsilon_decay}, target_freq={target_update_frequency}, goal_ratio={goal_ratio}')
    
    vocab = build_vocab(env)
    dqn = DQNLearner(
        state_encoder=make_encoder(vocab),
        input_dim=len(vocab),       # input_dim = len(vocab) = number of predicates
        num_actions=len(env.rules), # output_dim = num_actions = len(env.rules) = number of rules
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update_frequency=target_update_frequency,
        goal_ratio=goal_ratio
    )

    t4 = time.time()
    episode_rewards, episode_lengths = dqn.train(
        env=env,
        n_episodes=num_samples,
        max_steps=10000
    )
    t5 = time.time()
    
    if sweep_suffix:
        model_file = output_pref + f"-c-d-{sweep_suffix}.pt"
        vocab_file = output_pref + f"-c-v-{sweep_suffix}.json"
    else:
        model_file = output_pref + "-c-d.pt"
        vocab_file = output_pref + "-c-v.json"

    dqn.save(model_file)

    with open(vocab_file, 'w') as f:
        json.dump(vocab, f)

    print(f'[DQN]  Training time: {t5 - t4:.2f}s')
    
    if len(episode_rewards) > 0:
        success_count = sum(1 for r in episode_rewards if r > 1e-7)
        print(f'       Success episodes: {success_count}/{len(episode_rewards)}')
        print(f'       Success rate: {success_count / len(episode_rewards):.2%}')
        
        if success_count > 0:
            successful_lengths = [episode_lengths[i] for i in range(len(episode_rewards)) if episode_rewards[i] > 1e-7]
            print(f'       Successful episode steps: min={np.min(successful_lengths)}, max={np.max(successful_lengths)}, mean={np.mean(successful_lengths):.1f}')
        
        print(f'       Final avg reward (last 100): {(sum(episode_rewards[-100:]) / min(100, len(episode_rewards))):.2f}')
        print(f'       All episode steps: min={np.min(episode_lengths)}, max={np.max(episode_lengths)}, mean={np.mean(episode_lengths):.1f}')
    return dqn

if __name__ == "__main__":
    model_path   = sys.argv[1]
    init_term    = sys.argv[2]
    goal_prop    = sys.argv[3]
    num_samples  = int(sys.argv[4])
    output_pref  = sys.argv[5]

    trace_path = None
    sweep_mode = False
    
    learning_rate=5e-4
    gamma=0.95 
    tau=0.01
    epsilon_end=0.05
    epsilon_decay=0.0005
    target_update_frequency=50
    goal_ratio = 0.3

    if len(sys.argv) > 6 and sys.argv[6] == "sweep":
        sweep_mode = True
        learning_rate = float(sys.argv[7])
        gamma = float(sys.argv[8])
        tau = float(sys.argv[9])
        epsilon_end = float(sys.argv[10])
        epsilon_decay = float(sys.argv[11])
        target_update_frequency = int(sys.argv[12])
        goal_ratio = float(sys.argv[13])
        sweep_suffix = f"lr{learning_rate}-g{gamma}-t{tau}-e{epsilon_end}-d{epsilon_decay}-f{target_update_frequency}-g{goal_ratio}"
    elif len(sys.argv) > 6:
        trace_path = sys.argv[6]

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
        if not sweep_mode:
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
            dqn = run_dqn(
                learning_rate=learning_rate,
                gamma=gamma,
                tau=tau,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay,
                target_update_frequency=target_update_frequency,
                goal_ratio=goal_ratio,
                sweep_suffix=sweep_suffix if sweep_mode else None
            )
            compare_qtable_dqn(output_pref + '-c', dqn, m)
        sys.exit(0)

    if sweep_mode:
        envp = os.environ.copy()
        envp["MODE"] = "dqn"
        p = subprocess.run(
            [sys.executable] + sys.argv,
            env=envp, capture_output=True, text=True
        )
        if p.stdout: print(p.stdout, end="")
        if p.stderr: print(p.stderr, file=sys.stderr, end="")
        sys.exit(0)
    
    modes = []
    if trace_path is not None:
        modes.append("oracle")
    modes += ["cold", "dqn"]

    for mname in modes:
        envp = os.environ.copy()
        envp["MODE"] = mname
        args = [sys.executable, sys.argv[0], model_path, init_term, goal_prop, str(num_samples), output_pref]
        
        if trace_path:
            args.append(trace_path)

        p = subprocess.run(
            args,
            env=envp, capture_output=True, text=True
        )
        if p.stdout: print(p.stdout, end="")
        if p.stderr: print(p.stderr, file=sys.stderr, end="")