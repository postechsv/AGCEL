import maude
from AGCEL.AStar import Node, Search
from AGCEL.AStar import Node, Search
from AGCEL.MaudeEnv import MaudeEnv
from AGCEL.QLearning import QLearner
from AGCEL.Regression import RegressionScore

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

# ----- SEARCH WITHOUT TRAINING -----
print('\n=== SEARCH WITHOUT TRAINING ===')
init_term = m.parseTerm(init)
init_term.reduce()


n0 = Node(m, init_term)
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

    #res0[1].print_term()


#----- SEARCH WITH QTABLE -----
learner = QLearner()
learner.load_value_function(qtable_file, m)
V = learner.get_value_function()

print('\n=== SEARCH WITH QTABLE ===')
start_time = time.perf_counter()
res = Search().search(n0, V, 9999)
end_time = time.perf_counter()

print('[QTABLE] n_states:', res[2])
print(f'[QTABLE] Elapsed time: {(end_time - start_time)*1000:.3f} ms')
print('[QTABLE] n_states:', res[2])
print(f'[QTABLE] Elapsed time: {(end_time - start_time)*1000:.3f} ms')
if res[0]:
    print('[QTABLE] Goal reached!')
    # res[1].print_term()


# ----- SEARCH WITH REGRESSION -----
print('\n=== SEARCH WITH REGRESSION ===')

# Load data, regression model
reg = RegressionScore()
reg.train(qtable_file)

V_reg = reg.get_value_function()

start_time = time.perf_counter()
res_reg = Search().search(n0, V_reg, 9999)
end_time = time.perf_counter()

print('[REGRESSION] n_states:', res_reg[2])
print(f'[REGRESSION] Elapsed time: {(end_time - start_time)*1000:.3f} ms')
if res_reg[0]:
    print('[REGRESSION] Goal reached!')
    # res_reg[1].print_term()