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

env = MaudeEnv(m,prop,lambda : init,abst_mode='full')
learner = QLearner()
print(f'training {m}, {init} |= {prop} ... with {N} samples')
stat = learner.train(env, N)
print('qtable size :', learner.get_size())
learner.dump('qtable.maude', str(m))
print('dumped qtable : qtable.maude')
