import maude
from AGCEL.MaudeEnv import MaudeEnv
from AGCEL.QLearning import QLearner
from AGCEL.AStar import *
import time
import sys

model = sys.argv[1]
init = sys.argv[2]
prop = sys.argv[3]
qtable_file = sys.argv[4]

maude.init()
maude.load(model)
m = maude.getCurrentModule()

env = MaudeEnv(m, prop, lambda: init)

print('\n=== SEARCH WITHOUT TRAINING ===')
init_term = m.parseTerm(init)
init_term.reduce()
n0 = Node(m, init_term)

V0 = lambda node: 0

start_time = time.perf_counter()
res0 = Search().search(n0, V0, 9999)
end_time = time.perf_counter()

print('[BASELINE] n_states:', res0[2])
print(f'[BASELINE] Elapsed time: {(end_time - start_time)*1000:.3f} ms')
if res0[0]:
    print('[BASELINE] Goal reached!')
    res0[1].print_term()

### Load pretrained value function.
learner = QLearner()
learner.load_value_function(qtable_file, m)
V = learner.get_value_function()

print('\n=== SEARCH WITH TRAINED VALUE FUNCTION ===')
start_time = time.perf_counter()
res = Search().search(n0, V, 9999)
end_time = time.perf_counter()

print('[TRAINED] n_states:', res[2])
print(f'[TRAINED] Elapsed time: {(end_time - start_time)*1000:.3f} ms')
if res[0]:
    print('[TRAINED] Goal reached!')
    res[1].print_term()


### test script ###
# Trained on filter-3
# python3 test.py ./testcases/filter-3.maude init twoCrits trained/filter3-init3.agcel 
# python3 test.py ./testcases/filter-4.maude init twoCrits trained/filter3-init3.agcel 
# python3 test.py ./testcases/filter-5.maude init twoCrits trained/filter3-init3.agcel 

# Trained on filter-4
# python3 test.py ./testcases/filter-3.maude init twoCrits trained/filter4-init4.agcel 
# python3 test.py ./testcases/filter-4.maude init twoCrits trained/filter4-init4.agcel 
# python3 test.py ./testcases/filter-5.maude init twoCrits trained/filter4-init4.agcel 

# Trained on filter-5
# python3 test.py ./testcases/filter-3.maude init twoCrits trained/filter5-init5.agcel 
# python3 test.py ./testcases/filter-4.maude init twoCrits trained/filter5-init5.agcel 
# python3 test.py ./testcases/filter-5.maude init twoCrits trained/filter5-init5.agcel 