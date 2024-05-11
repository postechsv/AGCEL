import numpy as np
import heapq

from dp_env import *


# BFS with heuristics given by abstract Q table
def search(N=5, vec=None, max_step=10000, Qtable=None):
    if not Qtable is None:
        heuristics = True
    else:
        heuristics = False
    env = DiningPhilosEnv(N,vec)
    vec = env._get_vec()
    #abs_vec = env._get_abs_vec()
    #print('initial config:', vec)
    visited = set()

    i = 0
    queue = [(i,vec)] # (priority, vec)
    while not queue == []:
        vec = heapq.heappop(queue)[1]
        if vec in visited:
            continue
        i += 1
        visited.add(vec)
        env = DiningPhilosEnv(N, vec)
        abs_vec = env._get_abs_vec()
        if env.is_goal():
            #print('goal reached!')
            #print('vec:', vec)
            #print('num steps:', i)
            break
        elif i == max_step:
            #print('max step reached!')
            break
        nbrs = [(v, av) for (a, v, av) in env.next_actions if not v in visited] # unvisited next vecs
        if heuristics:
            p_nbrs = [(-Qtable[abs_vec + av], v) for (v, av) in nbrs] # prioritized nbrs
        else:
            p_nbrs = [(i, v) for (v, av) in nbrs] # prioritized nbrs
        #print(p_nbrs)
        for item in p_nbrs:
            heapq.heappush(queue, item)
    return i


