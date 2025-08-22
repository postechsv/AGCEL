import numpy as np
import torch

def extract_predicate_vector(obs_term):
    preds = []
    pred_container = list(obs_term.arguments())[0]
    def flatten(t):
        sym = str(t.symbol())
        if sym in ('_;_', 'and', '_`,_'):
            for a in t.arguments(): flatten(a)
        elif sym == '_:_' and len(list(t.arguments())) == 2:
            p, b = list(t.arguments())
            preds.append((str(p.symbol()), str(b.symbol()).lower() == 'true'))
    flatten(pred_container)
    return preds

def build_vocab(env):   # build predicate list
    return list({name for name, _ in extract_predicate_vector(env.get_obs()['state'])})

def make_encoder(vocab):
    idx = {n:i for i,n in enumerate(vocab)}
    dim = len(vocab)
    def encode(term):
        vec = np.zeros(dim, dtype=np.float32)
        for name, val in extract_predicate_vector(term):
            j = idx.get(name)
            if j is not None:
                vec[j] = 1.0 if val else 0.0
        return torch.from_numpy(vec)
    return encode