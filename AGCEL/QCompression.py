def extract_predicate_vector(obs_term):
    """Extract predicate names and their truth values from obs(...)"""

    preds = []
    pred_container = obs_term.arguments()[0]

    def flatten(t):
        sym = t.symbol().getName()
        if sym in ('_;_', 'and'):
            for arg in t.arguments():
                flatten(arg)
        else:
            pname = t.symbol().getName()
            val = t.arguments()[0].symbol().getName().lower() == 'true'
            preds.append((pname, val))

    return

def compress_qtable_pairwise():
    return

def infer_pairwise():
    return