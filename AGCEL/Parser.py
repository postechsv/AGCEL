import re

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

    # === DEBUG LOGS ===
    # print(f'[Parser] Parsed {len(trace)} transitions from "{file_path}"')
    # for idx, (s, a, ns) in enumerate(trace):
    #     print(f'  {idx:2d}: {a:10s} | {s[:60]} -> {ns[:60]}{" ..." if len(s)>60 or len(ns)>60 else ""}')

    return trace


def parse_agcel_line(line):
    state_match = re.search(r'<(.*)>', line)
    if not state_match:
        return None, None
    state_str = state_match.group(1)
    bool_values = [1 if 'true' in s else 0 for s in state_str.split(',')]
    score_match = re.search(r'\|->\s*([0-9.]+)', line)
    score = float(score_match.group(1)) if score_match else None
    return bool_values, score

def parse_agcel_file(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            if '<' in line and '|->' in line:
                vec, score = parse_agcel_line(line)
                if vec is not None and score is not None:
                    data.append((vec, score))
    return data