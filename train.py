import maude
from AGCEL.MaudeEnv import MaudeEnv
from AGCEL.QLearning import QLearner
import sys

#model = './benchmarks/onethirdrule/onethirdrule-hs.maude'
model = sys.argv[1]
init = sys.argv[2]
prop = sys.argv[3]
N = int(sys.argv[4])

maude.init()
maude.load(model)
m = maude.getCurrentModule()

env = MaudeEnv(m,prop,lambda : init)

### Train
learner = QLearner()
print(f'training {m}, {init} |= {prop} ... with {N} samples')
learner.train(env, N)
print('qtable size :', learner.get_size())
#learner.dump(f'qtable-{prop}.maude', str(m), prop)
learner.dump_value_function(f'value-function-{prop}.maude')


#learner.dump2(f'score-{prop}.maude', m, prop)
#print('dumped qtable : qtable.maude')


### Search right after training ###
from AGCEL.AStar import *

init = m.parseTerm('init3')
#init.reduce()
n0 = Node(m, init)
V = learner.get_value_function()
#res = Search().search(n0, V, 9999)

#if res[0]:
#    print('n_states:', res[2])
#    res[1].print_term()


