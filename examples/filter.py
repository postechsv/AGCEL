import sys
sys.path.append('../AGCEL')
# Assume this file is in the project root

import maude
from AGCEL.MaudeEnv import MaudeEnv
from AGCEL.QLearning import QLearner

model = './benchmarks/filter-analysis.maude'
init = 'init'
prop = 'twoCrits'

maude.init()
maude.load(model)
m = maude.getCurrentModule()

"""
Test for MaudeEnv
"""

# Test : MaudeEnv.__init__
print(type(m))
env = MaudeEnv(m,prop,lambda : init)

# Test : MaudeEnv.reset to init
env.reset()

# Test : MaudeEnv.get_obs
obs = env.get_obs()
print('G_state:', obs['G_state'])
print('state:', obs['state'])
print('actions:', obs['actions'])

# Test : MaudeEnv.reset to a particular state
env.reset(obs['G_state'])

# Test : MaudeEnv.step
env.step(obs['actions'][0])

# Test : MaudeEnv.obs
env.obs(obs['G_state'])

# Test : MaudeEnv.obs_act
[env.obs_act(label,sb) for label in env.rules for _, sb, _, _ in env.G_state.apply(label)]

# Test : MaudeEnv.get_rew
env.get_rew()

# Test : MaudeEnv.is_done
env.is_done()


"""
Test for QLearner
"""
# Test : QLearner.__init__
QL = QLearner()

# Test : QLearner.train
QL.train(env, 100)

# Test : QLearner.get_size
QL.get_size()

# Test : QLearner.dump_value_function
QL.dump_value_function('tmp')

# Test : QLearner.load_value_function
QL.load_value_function('tmp', m)
