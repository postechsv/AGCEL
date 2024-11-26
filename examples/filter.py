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

env = MaudeEnv(m,prop,lambda : init)


