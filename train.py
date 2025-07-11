import maude
from AGCEL.MaudeEnv import MaudeEnv
from AGCEL.QLearning import QLearner
import sys

# e.g. python3 train.py ./benchmarks/filter-analysis.maude init twoCrits 500 trained/filter-init3-twoCrits-500.agcel

model = sys.argv[1]
init = sys.argv[2]
prop = sys.argv[3]
N = int(sys.argv[4])
filename = sys.argv[5]

maude.init()
maude.load(model)
m = maude.getCurrentModule()

env = MaudeEnv(m,prop,lambda : init)

### Baseline: Search without training ###
# from AGCEL.AStar import *
# import time

# print('\n== SEARCH BEFORE TRAINING ===')
# init0 = m.parseTerm('init3')
# init0.reduce()
# n0 = Node(m, init0)

# V0 = lambda node: 0

# start_time = time.perf_counter()
# res0 = Search().search(n0, V0, 9999)
# end_time = time.perf_counter()

# if res0[0]:
#     print('[BASELINE] Goal reached!')
#     res0[1].print_term()
# else:
#     print('[BASELINE] Goal not found.')

# print('[BASELINE] n_states:', res0[2])
# elapsed_ms = (end_time - start_time) * 1000
# print(f'[BASELINE] Elapsed time: {elapsed_ms:.3f} ms')

### Train
print('\n=== TRAINING ===')
learner = QLearner()
print(f'TASK: Module({m}), Init({init}) |= Goal({prop})')
print(f'NUM_DATA: {N} samples')
learner.train(env, N)
print('\n=== RESULT ===')
print('qtable size :', learner.get_size())
#learner.dump(f'qtable-{prop}.maude', str(m), prop)
learner.dump_value_function(filename)


#learner.dump2(f'score-{prop}.maude', m, prop)
#print('dumped qtable : qtable.maude')


### After training ###
# print('\n=== SEARCH AFTER TRAINING ===')
# init = m.parseTerm('init3')
# init.reduce()
# n0 = Node(m, init)
# V = learner.get_value_function()

# start_time = time.perf_counter()
# res = Search().search(n0, V, 9999)
# end_time = time.perf_counter()

# if res[0]:
#     print('[TRAINED] Goal reached!')
#     res[1].print_term()
# else:
#     print('[TRAINED] Goal not found.')

# print('[TRAINED] n_states:', res[2])
# elapsed_ms = (end_time - start_time) * 1000
# print(f'[TRAINED] Elapsed time: {elapsed_ms:.3f} ms')

### Search right after training ###
# from AGCEL.AStar import *

# print('=== SEARCH ===')
# init = m.parseTerm('init3')
# init.reduce()
# n0 = Node(m, init)
# V = learner.get_value_function()
# res = Search().search(n0, V, 9999)

# if res[0]:
#    print('n_states:', res[2])
#    res[1].print_term()



###
# import maude
# from AGCEL.MaudeEnv import MaudeEnv
# from AGCEL.QLearning import QLearner
# import sys

# # python3 train.py ./testcases/filter-3.maude FILTER-INIT3 init twoCrits 500 trained/filter-init3-twoCrits-500.agcel
# # python3 train.py ./testcases/filter-4.maude FILTER-INIT4 init twoCrits 500 trained/filter-init4-twoCrits-500.agcel
# # python3 train.py ./testcases/filter-5.maude FILTER-INIT5 init twoCrits 500 trained/filter-init5-twoCrits-500.agcel

# # python3 train.py ./testcases/filter-3.maude FILTER-INIT3 init twoCrits 1000 trained/filter-init3-twoCrits-1000.agcel
# # python3 train.py ./testcases/filter-4.maude FILTER-INIT4 init twoCrits 1000 trained/filter-init4-twoCrits-1000.agcel
# # python3 train.py ./testcases/filter-5.maude FILTER-INIT5 init twoCrits 1000 trained/filter-init5-twoCrits-1000.agcel

# # python3 train.py ./testcases/filter-3.maude FILTER-INIT3 init twoCrits 1500 trained/filter-init3-twoCrits-1500.agcel
# # python3 train.py ./testcases/filter-4.maude FILTER-INIT4 init twoCrits 1500 trained/filter-init4-twoCrits-1500.agcel
# # python3 train.py ./testcases/filter-5.maude FILTER-INIT5 init twoCrits 1500 trained/filter-init5-twoCrits-1500.agcel

# model = sys.argv[1]
# module_name = sys.argv[2]
# init = sys.argv[3]
# prop = sys.argv[4]
# N = int(sys.argv[5])
# filename = sys.argv[6]

# maude.init()
# maude.load(model)
# m = maude.getModule(module_name)

# env = MaudeEnv(m, prop, lambda: init)

# print('\n=== TRAINING ===')
# learner = QLearner()
# print(f'TASK: Module({module_name}), Init({init}) |= Goal({prop})')
# print(f'NUM_DATA: {N} samples')
# learner.train(env, N)

# print('\n=== RESULT ===')
# print('qtable size :', learner.get_size())
# learner.dump_value_function(filename)