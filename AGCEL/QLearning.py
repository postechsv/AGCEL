import random
import numpy as np
import datetime
from tqdm import tqdm

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
        self.q_dict = dict()
        
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
        
    def argmax_q(self, s, actions): # nbrs: iterable if acfg's
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
    
    def dump(self, filename, module_name):
        f = open(filename, 'w')
        f.write(f'--- automatically generated at {datetime.datetime.now()}\n')
        f.write('mod QHS-SCORE is\n')
        f.write(f'  pr QHS-SCORE-BASE . pr {module_name} .\n')
        f.write('  var AS : HState . var AA : AAct .\n')
        q_dict = self.q_dict
        for s, d in q_dict.items():
            for a, q in d.items():
                f.write(f'  eq qtable({s}, {a}) = {q} [print "hit"] .\n')
        f.write(f'  eq qtable(AS, AA) = default [owise print "miss"] .\n')
        f.write('endm\n')
        f.close()
        
    def greedy_policy(self, obs):
        # returns -1 for error
        astate = obs["astate"]
        actions = obs["actions"]
        return self.argmax_q(astate,actions)
    
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
            
    def train(self, env, n_training_episodes):
        stat = 0
        for episode in tqdm(range(n_training_episodes)):
            # Reduce epsilon (because we need less and less exploration)
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

            obs = env.reset()
            step = 0
            done = False

            for step in range(max_steps):
                s = obs["astate"]
                a = self.eps_greedy_policy(obs, epsilon)

                # assert action not -1
                if type(a) == type(-1):
                    break

                obs, reward, done = env.step(a)
                ns = obs['astate']
                stat += reward

                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                nq = self.get_q(s, a) + learning_rate * (
                    reward + gamma * self.max_q(ns) - self.get_q(s, a) # FIXME!!!!! max_q(s')!!!!
                )
                self.set_q(s, a, nq)

                # If terminated or truncated finish the episode
                if done:
                    break

        print('training done!')
        return stat
