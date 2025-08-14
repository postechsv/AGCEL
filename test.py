import maude
from AGCEL.AStar import Node, Search
from AGCEL.MaudeEnv import MaudeEnv
from AGCEL.QLearning import QLearner

import time
import sys

# Usage: python3 test.py <maude_model> <init_term> <goal_prop> <qtable_file>
# python3 test.py testcases/filter-5.maude init twoCrits trained/filter-init3-twoCrits-500-o1.agcel

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


# ---- SEARCH WITHOUT TRAINING -----
V0 = lambda node: 0

print('\n=== SEARCH WITHOUT TRAINING ===')
start_time = time.perf_counter()
res0 = Search().search(n0, V0, 9999)
end_time = time.perf_counter()

print('[BASELINE] n_states:', res0[2])
print(f'[BASELINE] Elapsed time: {(end_time - start_time)*1000:.3f} ms')
if res0[0]:
    print('[BASELINE] Goal reached!')
    #res0[1].print_term()


# ----- SEARCH WITH QTABLE -----
print('\n=== SEARCH WITH QTABLE ===')
learner = QLearner()
learner.load_value_function(qtable_file, m)
V = learner.get_value_function()

start_time = time.perf_counter()
res = Search().search(n0, V, 9999)
end_time = time.perf_counter()

print('[QTABLE] n_states:', res[2])
print(f'[QTABLE] Elapsed time: {(end_time - start_time)*1000:.3f} ms')
if res[0]:
    print('[QTABLE] Goal reached!')
    # res[1].print_term()

# ----- SEARCH WITH Q_ABS -----
print('\n=== SEARCH WITH Q_ABS ===')
V_abs = learner.get_value_function_abs()

start_time = time.perf_counter()
res_abs = Search().search(n0, V_abs, 9999)
end_time = time.perf_counter()

print('[PA2] n_states:', res_abs[2])
print(f'[PA2] Elapsed time: {(end_time - start_time)*1000:.3f} ms')
if res_abs[0]:
    print('[PA2] Goal reached!')