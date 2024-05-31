import maude
from AGCEL.MaudeEnv import MaudeEnv
from AGCEL.QLearning import QLearner

model = './benchmarks/onethirdrule/onethirdrule-hs.maude'
maude.init()
maude.load(model)
m = maude.getCurrentModule()
print('Using', m, 'module')

env = MaudeEnv(m,'disagree',lambda : 'init3',abst_mode='full')
print(env.get_obs())

learner = QLearner()
print('training..')
stat = learner.train(env, 1000)
print(learner.get_size())
learner.dump('qtable.maude',str(m))
#print(learner.q_dict)

