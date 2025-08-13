import random
import numpy as np
import datetime
from tqdm import tqdm
from AGCEL.PA2 import QPatternCache

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
        
    def get_q(self, s, a):
        q_init = self.q_init
        if s in self.q_dict:
            return self.q_dict[s].get(a, q_init)
        return q_init
        
    def set_q(self, s, a, q):
        # TODO deepcopy terms
        if q == 0.0: # TODO
            return
        elif not s in self.q_dict:
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
    
    def dump_value_function(self, filename):
        with open(filename, 'w') as f:
            for s, _ in self.v_dict.items():
                f.write(f'{s} |-> {self.v_dict[s]}\n')

    def load_value_function(self, filename, m):
        self.v_dict = dict()
        with open(filename, 'r') as f:
            for line in f.readlines():
                state, value = line.split(" |-> ")
                state = m.parseTerm(state)
                state.reduce()
                value = float(value)
                self.v_dict[state] = value

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

    def pretrain(self, env, trace_path, repeat=100):
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

        #print(f'Oracle matched {matched//repeat}/{total//repeat} transitions ({100*matched/total:.1f}%)')
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

                if done:
                    break

        print('training done!')
        self.make_v_dict()