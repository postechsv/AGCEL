import maude
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
            pred_name = p.prettyPrint(maude.PRINT_WITH_PARENS)
            preds.append((pred_name, str(b.symbol()).lower() == 'true'))
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

def parse_qtable_file(filepath):
    entries = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if '|->' in line:
                key, val = line.rsplit('|->', 1)
                entries[key.strip()] = float(val.strip())
    return entries

def compare_qtable_dqn(qtable_file, dqn, m):
    from scipy.stats import spearmanr

    qtable = parse_qtable_file(qtable_file + '.agcel')
    if not qtable:
        print('[ALIGN] No QTable entries found')
        return

    q_vals, dqn_vals = [], []
    V_dqn = dqn.get_value_function(mode="dqn")

    for state_str, q_val in qtable.items():
        obs_term = m.parseTerm(state_str)
        if obs_term is None:
            continue
        obs_term.reduce()
        dqn_val = V_dqn(obs_term)
        q_vals.append(q_val)
        dqn_vals.append(dqn_val)
        
    if len(q_vals) < 2:
        print(f'[ALIGN] Only {len(q_vals)} entries')
        return
    
    corr, pval = spearmanr(q_vals, dqn_vals)

    print(f'\n=== QTABLE vs DQN ALIGNMENT ===')
    print(f'[ALIGN] Entries compared: {len(q_vals)}')
    print(f'[ALIGN] Rank correlation: {corr:.4f} (pval={pval:.4f})')