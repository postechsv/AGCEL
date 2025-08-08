import maude
from AGCEL.MaudeEnv import MaudeEnv
from AGCEL.QLearning import QLearner
from AGCEL.QCompression import compress_qtable_pairwise
import sys
import time

# Usage: python3 train.py <maude_model> <init_term> <goal_prop> <episode> <trace_path> <output_prefix>
# python3 train.py benchmarks/filter-analysis.maude init twoCrits 500 traces/filter-init3-twoCrits-1.trace trained/filter-init3-twoCrits-500
# python3 train.py benchmarks/filter-analysis.maude init twoCrits 500 traces/filter-init3-twoCrits-1.trace traces/filter-init3-twoCrits-2.trace traces/filter-init3-twoCrits-3.trace trained/filter-init3-twoCrits-500

model_path = sys.argv[1]
init_term = sys.argv[2]
goal_prop = sys.argv[3]
episode = int(sys.argv[4])
# trace_path = sys.argv[5]
# output_prefix = sys.argv[6]
trace_paths = sys.argv[5:-1] 
output_prefix = sys.argv[-1]

def make_suffix(trace_paths):
    idx = [tp.split("-")[-1].split(".")[0] for tp in trace_paths]
    return "-o" + "".join(idx)

trace_suffix = make_suffix(trace_paths)
warm_output_file = output_prefix + trace_suffix + '.agcel'

# === Setup ===
maude.init()
maude.load(model_path)
m = maude.getCurrentModule()
env = MaudeEnv(m, goal_prop, lambda: init_term)

print('\n=== TRAINING SETUP ===')
print(f'Module: {m}')
print(f'Init term: {init_term}')
print(f'Goal proposition: {goal_prop}')
print(f'Episodes: {episode}')
#print(f'Trace file: {trace_path}')
print(f'Trace files: {trace_paths}')
print(f'Output prefix: {output_prefix}')

# Warm-start (Oracle-pretrained) learner
print('\n=== WITH ORACLE ===')
warm_learner = QLearner()
#warm_learner.pretrain(env, trace_path)
warm_learner.pretrain_multi(env, trace_paths)
warm_size_before = warm_learner.get_size()
#print(f'Oracle QTable size (Before Training): {warm_size_before}')
t0 = time.time()
warm_learner.train(env, episode)
t1 = time.time()
warm_learner.dump_value_function(warm_output_file)

warm_compressed_q = compress_qtable_pairwise(warm_learner.q_dict)
print(f'[LOG] Compressed Q-table entries: {len(warm_compressed_q)}')
sample_keys = list(warm_compressed_q.keys())[:3]
for k in sample_keys:
    print(f'[LOG] Sample key: {k} -> {warm_compressed_q[k]:.4f}')


# Cold-start learner
print('\n=== WITHOUT ORACLE ===')
cold_learner = QLearner()
t2 = time.time()
cold_learner.train(env, episode)
t3 = time.time()


# Result
print('\n=== SUMMARY ===')
print(f'[Warm] Training time: {t1 - t0:.2f}s, # Entries: {warm_size_before} -> {warm_learner.get_size()}')
print(f'       Value function: {warm_output_file.split('/')[-1]}')
print(f'[Cold] Training time: {t3 - t2:.2f}s, # Entries: {cold_learner.get_size()}')