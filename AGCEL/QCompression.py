def extract_predicate_vector(obs_term):
    """Extract predicate names and their truth values from obs(...)"""

    pred_container = obs_term.arguments()[0]

    def flatten(t):
        sym = t.symbol().getName()
        if sym in ('_;_', 'and'):
            for arg in t.arguments():
                flatten(arg)

    return

def compress_qtable_pairwise():
    return

def infer_pairwise():
    return