import maude
from AGCEL.MaudeEnv import MaudeEnv
from AGCEL.QLearning import QLearner
from AGCEL.DQNLearning import DQNLearner
from AGCEL.common import build_vocab, make_encoder
import sys
import time
import json

# Usage: python3 train.py <maude_model> <init_term> <goal_prop> <num_samples> <output_file_prefix> <trace_path>
# python3 train.py benchmarks/filter-analysis.maude init twoCrits 500 trained/filter-init3-twoCrits-500 traces/filter-init3-twoCrits-1.trace

model_path = sys.argv[1]
init_term = sys.argv[2]
goal_prop = sys.argv[3]
num_samples = int(sys.argv[4])
output_prefix = sys.argv[5]
if len(sys.argv) == 6:
    trace_path = None
elif len(sys.argv) == 7:
    trace_path = sys.argv[6]

# === Setup ===
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
print(f'Output prefix: {output_prefix}')

if trace_path is not None:
    # === Extract suffix from trace file name ===
    trace_suffix = "-o" + trace_path.split("-")[-1].split(".")[0] if "-" in trace_path else "-oracle"
    oracle_output_file = output_prefix + trace_suffix + '.agcel'

    # Oracle-pretrained learner
    print('\n=== [WITH ORACLE] ===')
    learner_oracle = QLearner()
    t0 = time.time()
    learner_oracle.pretrain(env, trace_path)
    oracle_size_before = learner_oracle.get_size()
    #print(f'Oracle QTable size (Before Training): {oracle_size_before}')
    learner_oracle.train(env, num_samples)
    t1 = time.time()
    learner_oracle.dump_value_function(oracle_output_file)
    #print(f'Oracle QTable size (After Training): {learner_oracle.get_size()}')
    #print(f'Oracle training time: {t1 - t0:.2f}s')
    #print(f'Output: {oracle_output_file.split('/')[-1]}')

# Cold-start learner
cold_output_file = output_prefix + "-c" + '.agcel'
print('\n=== [WITHOUT ORACLE] ===')
learner_cold = QLearner()
t2 = time.time()
learner_cold.train(env, num_samples)
t3 = time.time()
learner_cold.dump_value_function(cold_output_file)
#print(f'Cold QTable size: {learner_cold.get_size()}')
#print(f'Cold training time: {t3 - t2:.2f}s')
#print(f'Output: {cold_output_file.split('/')[-1]}')


# === DQN ===
print('\n=== [DQN] ===')
vocab = build_vocab(env)

dqn = DQNLearner(env, state_encoder=make_encoder(vocab), input_dim=len(vocab), num_actions=len(env.rules), gamma=0.95, lr=1e-3, tau=0.01)
t4 = time.time()
dqn.train(n_training_episodes=num_samples)
t5 = time.time()

dqn_model_file = output_prefix + '-dqn.pt'
dqn_vocab_file = output_prefix + '-dqn-vocab.json'
dqn.save_model(dqn_model_file)
with open(dqn_vocab_file, 'w') as f:
    json.dump(vocab, f)


# === Result ===
print('\n=== SUMMARY ===')
if trace_path is not None:
    print(f'[Warm] Training time: {t1 - t0:.2f}s, # Entries: {oracle_size_before} -> {learner_oracle.get_size()}')
    print(f'       Value function: {oracle_output_file.split("/")[-1]}')
else:
    print('[Warm] Skipped (no trace)')
print(f'[Cold] Training time: {t3 - t2:.2f}s, # Entries: {learner_cold.get_size()}')
print(f'[DQN]  Training time: {t5 - t4:.2f}s')
print(f'       Model: {dqn_model_file.split("/")[-1]}')
print(f'       Vocab: {dqn_vocab_file.split("/")[-1]}')