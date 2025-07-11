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
    # print(f'[TraceParser] Parsed {len(trace)} transitions from "{file_path}"')
    # for idx, (s, a, ns) in enumerate(trace):
    #     print(f'  {idx:2d}: {a:10s} | {s[:60]} -> {ns[:60]}{" ..." if len(s)>60 or len(ns)>60 else ""}')

    return trace