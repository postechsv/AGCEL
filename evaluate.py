import numpy as np
from tqdm import tqdm
from search import *

def evaluate(num_samples, N, max_steps, Qtable):
    res_fifo = []
    res_qlmc = []
    for i in tqdm(range(num_samples)):
        env = DiningPhilosEnv(N)
        vec = env._get_vec()
        #print('----------')
        #print('initial config:', vec)
        num_states_fifo = search(N, vec, max_steps)
        res_fifo.append(num_states_fifo)
        num_states_qlmc = search(N, vec, max_steps, Qtable)
        res_qlmc.append(num_states_qlmc)
        #print('fifo:', num_states_fifo, ', qlmc:', num_states_qlmc)
    avg_fifo = np.average(np.array(res_fifo))
    avg_qlmc = np.average(np.array(res_qlmc))
    print('=== STAT ===')
    print('N:', N)
    print('Qtable density:', np.count_nonzero(Qtable), '/', Qtable.size)
    print('num_samples:', num_samples)
    print('max_steps:', max_steps)
    print('avg_fifo:', avg_fifo)
    print('avg_qlmc:', avg_qlmc)

Qtable = np.load('trained_abs_qtable.npy')
evaluate(30, 5, 10000, Qtable)
