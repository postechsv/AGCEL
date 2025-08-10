import maude
from AGCEL.MaudeEnv import MaudeEnv
from AGCEL.QLearning import QLearner
import sys
import time

# Usage: python3 train.py <maude_model> <init_term> <goal_prop> <episode> <trace_path> <output_prefix>
# python3 train.py benchmarks/filter-analysis.maude init twoCrits 500 traces/filter-init3-twoCrits-1.trace trained/filter-init3-twoCrits-500

model_path = sys.argv[1]
init_term = sys.argv[2]
goal_prop = sys.argv[3]
episode = int(sys.argv[4])
trace_path = sys.argv[5]
output_prefix = sys.argv[6]

trace_suffix = "-o" + trace_path.split("-")[-1].split(".")[0] if "-" in trace_path else "-oracle"
warm_output_file = output_prefix + trace_suffix + '.agcel'
#cold_output_file = output_prefix + "-cold" + trace_suffix[2:] + ".agcel"

# === Setup ===
maude.init()
maude.load(model_path)
maude.load(model_path)
m = maude.getCurrentModule()
env = MaudeEnv(m, goal_prop, lambda: init_term)

print('\n=== TRAINING SETUP ===')
print(f'Module: {m}')
print(f'Init term: {init_term}')
print(f'Goal proposition: {goal_prop}')
print(f'Episodes: {episode}')
print(f'Trace file: {trace_path}')
print(f'Trace file: {trace_path}')
print(f'Output prefix: {output_prefix}')

# Warm-start (Oracle-pretrained) learner
print('\n=== WITH ORACLE ===')
warm_learner = QLearner()
warm_learner.pretrain(env, trace_path)
warm_size_before = warm_learner.get_size()
t0 = time.time()
warm_learner.train(env, episode)
t1 = time.time()
warm_learner.dump_value_function(warm_output_file)
#print(f'Oracle QTable size (After Training): {warm_learner.get_size()}')
#print(f'Oracle training time: {t1 - t0:.2f}s')
#print(f'Output: {warm_output_file.split('/')[-1]}')

# print(f'[LOG] Raw Q-table entries: {count_qtable_entries(warm_learner.q_dict)}')
# warm_compressed_q = compress_qtable_pairwise(warm_learner.q_dict)
# print(f'[LOG] Compressed Q-table entries: {len(warm_compressed_q)}')
# sample_keys = list(warm_compressed_q.keys())
# for k in sample_keys:
#     print(f'[LOG] Sample key: {k} -> {warm_compressed_q[k]:.4f}')


# Cold-start learner
print('\n=== WITHOUT ORACLE ===')
cold_learner = QLearner()
t2 = time.time()
cold_learner.train(env, episode)
t3 = time.time()
#cold_learner.dump_value_function(cold_output_file)
#print(f'Cold QTable size: {cold_learner.get_size()}')
#print(f'Cold training time: {t3 - t2:.2f}s')
#print(f'Output: {cold_output_file.split('/')[-1]}')

# Result
print('\n=== SUMMARY ===')
print(f'[Warm] Training time: {t1 - t0:.2f}s, # Entries: {warm_size_before} -> {warm_learner.get_size()}')
print(f'       Value function: {warm_output_file.split('/')[-1]}')
print(f'[Cold] Training time: {t3 - t2:.2f}s, # Entries: {cold_learner.get_size()}')
#print(f'       Value function: {cold_output_file.split('/')[-1]}')