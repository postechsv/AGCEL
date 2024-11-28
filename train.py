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

### Train
print('=== TRAINING ===')
learner = QLearner()
print(f'TASK: Module({m}), Init({init}) |= Goal({prop})')
print(f'NUM_DATA: {N} samples')
learner.train(env, N)
print('=== RESULT ===')
print('qtable size :', learner.get_size())
#learner.dump(f'qtable-{prop}.maude', str(m), prop)
learner.dump_value_function(filename)


#learner.dump2(f'score-{prop}.maude', m, prop)
#print('dumped qtable : qtable.maude')


### Search right after training ###
from AGCEL.AStar import *

#init = m.parseTerm('init3')
#init.reduce()
#n0 = Node(m, init)
#V = learner.get_value_function()
#res = Search().search(n0, V, 9999)

#if res[0]:
#    print('n_states:', res[2])
#    res[1].print_term()


