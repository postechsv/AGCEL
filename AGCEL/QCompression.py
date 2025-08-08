def extract_predicate_vector(obs_term):
    """Extract predicate names and their truth values from obs(...)"""

    preds = []
    pred_container = list(obs_term.arguments())[0]

    def flatten(t):
        sym = str(t.symbol())
        if sym in ('_;_', 'and'):
            for arg in t.arguments():
                flatten(arg)
        else:
            pname = str(t.symbol())
            val = str(list(t.arguments())[0].symbol()).lower() == 'true'
            preds.append((pname, val))

    flatten(pred_container)
    return preds

def compress_qtable_pairwise():
    return

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