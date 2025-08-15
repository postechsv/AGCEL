import random
import numpy as np
import datetime
from tqdm import tqdm
from itertools import combinations

# Training parameters
learning_rate = 0.7  # Learning rate

# Environment parameters
max_steps = 300  # Max steps per episode
gamma = 0.95  # Discounting rate

# Exploration parameters
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.0005  # Exponential decay rate for exploration prob

class QLearner():
    def __init__(self):
        self.q_init = 0.0
        self.q_dict = dict() # score(s,a)
        self.v_dict = dict()
        self.scores = dict() # score(p,q)
        self.q_abs = dict()         # score(s_abs, a)
        self.abs_mask_sizes = (1,)  # number of masking bits (default: 1)
        
    # obs term to a boolean list
    def parse_obs(self, obs_term):
        # extract predicate names and their truth values from obs(...)
        def extract_predicate_vector(obs_term):
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

            flatten(pred_container)
            #print(f'[LOG] Final predicate vector: {preds}')
            return preds

        pairs = extract_predicate_vector(obs_term)  # [('p1', True), ('p2', False), ('p3', True)]
        # names = [name for name, _ in pairs]             # ['p1', 'p2', 'p3']
        # m = {name: int(val) for name, val in pairs}     # {'p1': 1, 'p2': 0, 'p3': 1}
        # vec = tuple(m.get(name, 0) for name in names)   # (1, 0, 1)
        # return vec, names
        return [1 if b else 0 for _, b in pairs]
    
    # masks m bits (dim=n); keeps n-m bits
    def keep_idx(self, n):
        keeps = set()
        for m in self.abs_mask_sizes:
            k = n - m
            if k < 0:
                continue
            for keep in combinations(range(n), k):
                keeps.add(tuple(keep))
        return keeps

    # update q_abs: when updating Q(s,a)=q, max all s_abs of s
    def set_q_abs(self, s, a, q):
        vals = self.parse_obs(s)
        n = len(vals)

        for keep in self.keep_idx(n):
            keep_vals = tuple(vals[i] for i in keep)
            s_abs = (keep, keep_vals)
            if s_abs not in self.q_abs:
                self.q_abs[s_abs] = { a : q }
            else:
                if q > self.q_abs[s_abs].get(a, self.q_init):
                    self.q_abs[s_abs][a] = q

    def get_q(self, s, a):
        q_init = self.q_init
        if s in self.q_dict:
            return self.q_dict[s].get(a, q_init)
        return q_init
        
    def set_q(self, s, a, q):
        # TODO deepcopy terms
        if q == 0.0: # TODO
            return
        elif s not in self.q_dict:
            self.q_dict[s] = { a : q }
        else:
            self.q_dict[s][a] = q
        
    def argmax_q(self, s, actions):
        q_dict = self.q_dict
        if s in q_dict and len(actions) != 0:
            d = { a : q_dict[s].get(a, self.q_init) for a in actions } # d = restriction of q_dict to tl
            return max(d, key=d.get) # FIXME: random choice if tie
        else:
            return -1
        
    def max_q(self, s):
        q_dict = self.q_dict
        if s in q_dict: # assume q_dict[t] is nonempty
            return max(q_dict[s].values())
        return self.q_init
    
    def get_size(self):
        # returns the number of nonzero entries in the QTable
        ret = 0
        for _, d in self.q_dict.items():
            ret += len(d)
        return ret
    
    def print_v(self):
        q_dict = self.q_dict
        print(f'fmod SCORE is')
        for t in q_dict:
            print(f'  eq score({t}) = {self.max_q(t)} .')
        print(f'  eq score(X) = {self.q_init} [owise] .')
        print(f'endfm')        
    
    def dump(self, filename, module_name, goal):
        f = open(filename, 'w')
        f.write(f'--- automatically generated at {datetime.datetime.now()}\n')
        f.write('mod QTABLE is\n')
        f.write(f'  pr QTABLE-BASE . pr {module_name} .\n')
        f.write('  var S : MDPState . var A : MDPAct .\n')
        q_dict = self.q_dict
        for s, d in q_dict.items():
            for a, q in d.items():
                f.write(f'  eq q({goal}, {s}, {a}) = {q} [print "hit"] .\n')
        f.write(f'  eq q({goal}, S, A) = bot [owise print "miss"] .\n')
        f.write('endm\n')
        f.close()

    def make_v_dict(self):
        self.v_dict = dict()
        for s, _ in self.q_dict.items():
            self.v_dict[s] = self.max_q(s)

    def get_value_function(self):
        return (lambda s : self.v_dict.get(s, self.q_init))
    
    # PA2: V(s) = max_{s' in matching(s)} { max_a Q(s', a) }
    def get_value_function_abs(self):
        if not self.q_abs:
            return (lambda _: self.q_init)

        idx = {}
        for (keep, keep_vals), by_act in self.q_abs.items():
            d = idx.setdefault(keep, {})
            d[keep_vals] = max(by_act.values()) if by_act else self.q_init

        cache = {}
        
        def V_abs(obs_term):
            key = obs_term.prettyPrint(0)
            if key in cache:
                return cache[key]
            
            vals = self.parse_obs(obs_term)
            best = self.q_init

            for (keep, keep_vals), a in self.q_abs.items(): # s_abs = (keep, keep_vals)
                proj = tuple(vals[i] for i in keep) # projection of keep idx
                if proj != keep_vals:
                    continue
                cand = max(a.values()) if a else self.q_init  # max among max_a Q per action
                if cand > best:
                    best = cand

            return best

        return V_abs

    def dump_value_function(self, filename):
        with open(filename, 'w') as f:
            for s, _ in self.v_dict.items():
                f.write(f'{s} |-> {self.v_dict[s]}\n')

    def dump_abs_table(self, filename):
        with open(filename, 'w') as f:
            for (keep, keep_vals), by_act in self.q_abs.items():
                max_q = max(by_act.values()) if by_act else self.q_init
                f.write(
                    # keep=[...], keep_vals=[...] |-> <max_q> ; { 'act1':q1, 'act2':q2, ... }
                    f'keep={list(keep)}, keep_vals={list(keep_vals)} |-> {max_q} ; {by_act}\n'
                )

    def load_value_function(self, filename, m):
        self.v_dict = dict()
        with open(filename, 'r') as f:
            for line in f.readlines():
                state, value = line.split(" |-> ")
                state = m.parseTerm(state)
                state.reduce()
                value = float(value)
                self.v_dict[state] = value

    def load_abs_table(self, filename):
        self.q_abs = {}
        with open(filename) as f:
            for line in f:
                left, right = line.split("|->")
                max_q, a_dict = right.split(";")
                keep_str = left.split("keep=")[1].split(", keep_vals=")[0].strip()
                keep_vals_str = left.split("keep_vals=")[1].strip()
                keep = tuple(int(x) for x in keep_str.strip("[]").split(",") if x.strip()!="")
                keep_vals = tuple(int(x) for x in keep_vals_str.strip().strip("[]").split(",") if x.strip()!="")
                by_act = {}
                if "{" in a_dict:
                    for max_q in a_dict.split("{",1)[1].rsplit("}",1)[0].strip().split(","):
                        a,q = max_q.split(":")
                        by_act[a.strip().strip("'")] = float(q)
                self.q_abs[(keep, keep_vals)] = by_act

    def greedy_policy(self, obs):
        # returns -1 for error
        state = obs["state"]
        actions = obs["actions"]
        return self.argmax_q(state,actions)
    
    def eps_greedy_policy(self, obs, epsilon):
        # returns -1 for error
        r = random.uniform(0, 1)
        if r > epsilon: # exploitation
            return self.greedy_policy(obs)
        else: # exploration
            actions = obs["actions"]
            if len(actions) != 0:
                return random.choice(actions)
            else:
                return -1

    def oracle_policy(self, s, ns, actions, env):
        for a in actions:
            for next_state, act in env.nbrs:
                if act == a:
                    obs_ns = env.obs(next_state)
                    if obs_ns == ns:
                        return a
        return -1

    def pretrain(self, env, trace_path, repeat=10):
        from AGCEL.Parser import parse_trace

        trace = parse_trace(trace_path)
        matched = 0
        total = 0

        for _ in range(repeat):
            for i in range(len(trace)):
                s_str, _, ns_str = trace[i]
                s_term = env.m.parseTerm(s_str)
                ns_term = env.m.parseTerm(ns_str)
                s_term.reduce()
                ns_term.reduce()
                env.reset(to_state=s_term)
                obs_s = env.get_obs()
                s = obs_s['state']
                ns = env.obs(ns_term)
                a = self.oracle_policy(s, ns, obs_s['actions'], env)
                total += 1
                if isinstance(a, int) and a == -1:
                    continue
                matched += 1

                reward_term = env.m.parseTerm(f'reward({ns.prettyPrint(0)})')
                reward_term.reduce()
                r = reward_term.toFloat()

                if i < len(trace) - 1:
                    _, _, next_ns_str = trace[i + 1]
                    next_s_term = env.m.parseTerm(next_ns_str)
                    next_s_term.reduce()
                    next_ns = env.obs(next_s_term)
                    max_next_q = self.max_q(next_ns)
                else:
                    max_next_q = 0.0

                q = self.get_q(s, a)
                nq = q + learning_rate * (r + gamma * max_next_q - q)
                self.set_q(s, a, nq)
                self.set_q_abs(s, a, nq)

        print(f'Oracle matched {matched//repeat}/{total//repeat} transitions ({100*matched/total:.1f}%)')
        self.make_v_dict()
        

    def train(self, env, n_training_episodes):
        for episode in tqdm(range(n_training_episodes)):
            # Reduce epsilon (because we need less and less exploration)
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

            obs = env.reset()
            done = False

            for _ in range(max_steps):
                s = obs["state"]
                a = self.eps_greedy_policy(obs, epsilon)

                # assert action not -1
                if type(a) == type(-1):
                    break

                obs, reward, done = env.step(a)
                ns = obs['state']

                # Update Q(s,a) := Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                nq = self.get_q(s, a) + learning_rate * (
                    reward + gamma * self.max_q(ns) - self.get_q(s, a)
                )
                self.set_q(s, a, nq)
                self.set_q_abs(s, a, nq)

                if done:
                    break

        print('training done!')
        self.make_v_dict()