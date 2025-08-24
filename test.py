import maude
from AGCEL.MaudeEnv import MaudeEnv
from AGCEL.QLearning import QLearner
from AGCEL.AStar import *
from AGCEL.AStar import *
from AGCEL.DQNLearning import DQNLearner
from AGCEL.common import make_encoder
import json, re, sys, time

# Usage: python3 test.py <maude_model> <init_term> <goal_prop> <qtable_file>
# python3 test.py testcases/filter-5.maude init twoCrits trained/filter-init3-twoCrits-500-o1

model = sys.argv[1]
init = sys.argv[2]
prop = sys.argv[3]
qtable_file = sys.argv[4]

maude.init()
maude.load(model)
m = maude.getCurrentModule()

env = MaudeEnv(m, prop, lambda: init)

init_term = m.parseTerm(init)
init_term.reduce()
n0 = Node(m, init_term)

#V0 = lambda node: 0
V0 = lambda obs_term, g_state=None: 0

print('\n=== SEARCH WITHOUT TRAINING ===')
start_time = time.perf_counter()
res0 = Search().search(n0, V0, 9999)
end_time = time.perf_counter()

print('[BASELINE] n_states:', res0[2])
print(f'[BASELINE] Elapsed time: {(end_time - start_time)*1000:.3f} ms')
if res0[0]:
    print('[BASELINE] Goal reached!')

# Load pretrained value function
learner = QLearner()
learner.load_value_function(qtable_file + '.agcel', m)
V = learner.get_value_function()

print('\n=== SEARCH WITH TRAINED VALUE FUNCTION ===')
start_time = time.perf_counter()
res = Search().search(n0, V, 9999)
end_time = time.perf_counter()

print('[TRAINED] n_states:', res[2])
print(f'[TRAINED] Elapsed time: {(end_time - start_time)*1000:.3f} ms')
if res[0]:
    print('[TRAINED] Goal reached!')


m = re.search(r'(.+?)(-o\d+|-oracle)?$', qtable_file)
base_prefix = m.group(1) if m else qtable_file

dqn_model_file = base_prefix + '-dqn.pt'
dqn_vocab_file = base_prefix + '-dqn-vocab.json'

with open(dqn_vocab_file, 'r') as f:
    vocab = json.load(f)
dqn = DQNLearner(env, state_encoder=make_encoder(vocab), input_dim=len(vocab),
                 num_actions=len(env.rules), gamma=0.95, lr=1e-3, tau=0.01)
dqn.load_model(dqn_model_file)
V_dqn = dqn.get_value_function()

print('\n=== SEARCH WITH DQN VALUE FUNCTION ===')
start_time = time.perf_counter()
res_dqn = Search().search(n0, V_dqn, 9999)
end_time = time.perf_counter()

print('[DQN] n_states:', res_dqn[2])
print(f'[DQN] Elapsed time: {(end_time - start_time)*1000:.3f} ms')
if res_dqn[0]:          
    print('[DQN] Goal reached!')