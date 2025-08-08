from collections import defaultdict
from itertools import combinations
import numpy as np

def extract_predicate_vector(obs_term):
    """Extract predicate names and their truth values from obs(...)"""

    preds = []
    pred_container = list(obs_term.arguments())[0]

    def flatten(t):
        sym = str(t.symbol())
        if sym in ('_;_', 'and', '_`,_'):
            for arg in t.arguments():
                flatten(arg)
        elif sym == '_:_' and len(list(t.arguments())) == 2:
            pred_term = list(t.arguments())[0]
            bool_term = list(t.arguments())[1]
            pname = str(pred_term.symbol())
            val = str(bool_term.symbol()).lower() == 'true'
            preds.append((pname, val))
            print(f'[LOG] Found predicate: {pname} = {val}')
        else:
            print(f'[LOG] Ignored term: {t.prettyPrint(0)} (symbol: {sym})')

    flatten(pred_container)
    return preds

def compress_qtable_pairwise(q_dict):
    pairwise_q = defaultdict(list)

    for state_term, action_dict in q_dict.items():
        pred_vector = extract_predicate_vector(state_term)
        for (p1, v1), (p2, v2) in combinations(pred_vector, 2):
            for action, q_val in action_dict.items():
                key = ((p1, v1), (p2, v2), str(action))
                pairwise_q[key].append(q_val)

    return {k: np.mean(v) for k, v in pairwise_q.items()}

def infer_pairwise():
    return

if __name__ == '__main__':
    import maude
    from MaudeEnv import MaudeEnv

    maude.init()
    maude.load('benchmarks/filter-analysis.maude')
    m = maude.getCurrentModule()

    init_term = 'init'
    goal = 'twoCrits'
    env = MaudeEnv(m, goal, lambda: init_term)

    obs = env.get_obs()
    obs_term = obs["state"]
    print(f'[LOG] Raw obs term: {obs_term.prettyPrint(0)}')

    vec = extract_predicate_vector(obs_term)
    print(f'[LOG] Extracted predicate vector: {vec}')