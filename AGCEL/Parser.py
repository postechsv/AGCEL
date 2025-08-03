import re

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