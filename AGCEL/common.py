import numpy as np
import torch

def extract_predicate_vector(obs_term): # [("critLHS", True), ("waitLHS", False)]
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

def build_vocab(env):   # ["critLHS", "waitLHS"]
    return list({name for name, _ in extract_predicate_vector(env.get_obs()['state'])})

def make_encoder(vocab):
    idx = {n:i for i,n in enumerate(vocab)}
    dim = len(vocab)
    def encode(term):   # [1.0, 0.0]
        vec = np.zeros(dim, dtype=np.float32)
        for name, val in extract_predicate_vector(term):
            j = idx.get(name)
            if j is not None:
                vec[j] = 1.0 if val else 0.0
        return torch.from_numpy(vec)
    return encode

def parse_trace(file_path):
    trace = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    states = []
    actions = []

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()

        if line.startswith("state") and "Conf:" in line:    # state
            conf = line.split("Conf:", 1)[1].strip()
            conf = conf.replace(") ;", ");")
            states.append(conf)
            i += 1

        elif line.startswith("===["):                       # transition
            action_lines = []
            while i < n and "===>" not in lines[i]:
                action_lines.append(lines[i].strip())
                i += 1
            if i < n:
                action_lines.append(lines[i].strip())
                i += 1
            action_str = ' '.join(action_lines)
            if "[label" in action_str:
                label = action_str.split("[label", 1)[1].split("]")[0].strip()
                actions.append(label)
        else:
            i += 1

    for j in range(min(len(actions), len(states) - 1)):
        trace.append((states[j], actions[j], states[j + 1]))
        
    return trace

def compare_qtable_dqn(qtable_learner, dqn, env):
    q_values = qtable_learner.q_table