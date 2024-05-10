from gym import Env
from gym import spaces
import numpy as np
import random
import copy

class DiningPhilosEnv(Env):
    def __init__(self, N, vec=None):
        self.N = N # number of philosophers = number of chopsticks
        
        if vec != None:
            assert len(vec) == 2*N, "The length of vec should be equal to N."
            self.last_action = -1
            self.P = np.array(vec[:N])
            self.C = np.array(vec[N:])
            self.next_actions = self._get_next_actions()
        else:
            self.reset()
        
        self.observation_space = spaces.Dict(
            {
                "P": spaces.Box(0, 3, shape=(N,), dtype=int),
                "C": spaces.Box(0, 1, shape=(N,), dtype=int),
            }
        )
        self.action_space = spaces.Discrete(4*N)
        
        self.shape_state_space = ((4,)* self.N) + ((2,)* self.N)
        self.shape_action_space = (4 * self.N,)
        
    def step(self, a):
        self._step(a)
        self.next_actions = self._get_next_actions()
        observation = self._get_obs()
        info = self._get_info()
        reward = 0
        done = False
        if self.is_goal():
            # deadlock
            reward = 1
            done = True
        return observation, reward, done, info
    
    def _step(self, a):
        # to avoid recursion. can be thought of as incomplete self.step()
        self.last_action = a
        phil_id = a // 4
        action_type = a % 4
        
        if action_type == 0: # think -> hungry
            self.P[phil_id] = 1
        elif action_type == 1: # pick left
            self.P[phil_id] += 1
            self.C[phil_id] = 0
        elif action_type == 2: # pick right
            self.P[phil_id] += 1
            self.C[(phil_id + 1) % self.N] = 0
        else: # eat -> think
            self.P[phil_id] = 0
            self.C[phil_id] = 1
            self.C[(phil_id + 1) % self.N] = 1
        
        vec = self._get_vec()
        abs_vec = self._get_abs_vec()
        return vec, abs_vec
    
    def is_goal(self):
        return not np.any(self._get_mask()) # deadlock
    
    def _get_obs(self):
        return {
            "vec": self._get_vec(), #tuple(self.P) + tuple(self.C),
            "abs_vec": self._get_abs_vec(),
            "mask": self._get_mask(),
            "abs_mask": self._get_abs_mask(),
        }
    
    def _get_info(self):
        return {
            "action_desc" : self._get_action_desc()
        }
    
    def reset(self):
        self.last_action = -1
        self._generate_random_state()
        self.next_actions = self._get_next_actions()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def _get_mask(self):
        # 0: disabled, 1: enabled
        N = self.N
        P = self.P
        C = self.C
        mask = np.zeros(4 * N, dtype=np.int8)
        for phil_id in range(N):
            if P[phil_id] == 0:
                mask[4 * phil_id + 0] = 1
            elif P[phil_id] == 3:
                mask[4 * phil_id + 3] = 1
            else:
                mask[4 * phil_id + 1] = C[phil_id]
                mask[4 * phil_id + 2] = C[(phil_id + 1) % self.N]
        return mask
    
    def _get_abs_mask(self):
        # 0: disabled, 1: enabled
        abs_mask = np.zeros(shape=(2,2,2,2), dtype=np.int8)
        next_actions = self._get_next_actions()
        for _, _, abs_a in next_actions:
            abs_mask[abs_a] = 1
        return abs_mask
    
    def abs_step(self, abs_action):
        # from abstract action, step with a randomly chosen corresponding concrete action
        # assume abs_action is available i.e. there exists at least one concrete action
        actions = [a for (a, _, aa) in self.next_actions if aa == abs_action]
        action = random.choice(actions)
        return self.step(action)
    
    def _get_next_actions(self):
        # TODO: refactoring pythonic
        mask = self._get_mask()
        next_actions = []
        for action, flag in enumerate(mask):
            if flag == 1:
                vec, abs_vec = copy.deepcopy(self)._step(action) # garbage collection?
                next_actions.append((action, vec, abs_vec))
        return next_actions
                
        
    def _get_action_desc(self):
        # returns the description of the last action
        a = self.last_action
        if a == -1:
            return "no action"
        phil_id = a // 4
        action_type = a % 4
        desc = "philsopher " + str(phil_id)
        if action_type == 0: # think -> hungry
            desc += " gets hungry"
        elif action_type == 1: # pick left
            desc += " picks left chopstick"
        elif action_type == 2: # pick right
            desc += " picks right chopstick"
        else: # eat -> think
            desc += " stops eating"
        return desc
    
    def _generate_random_state(self):
        N = self.N
        self.P = np.zeros(N, dtype=int)
        self.C = np.ones(N, dtype=int)
        
        for i in range(N):
            c = random.randrange(3)
            if c == 1:
                # to left
                self.P[(i-1) % N] += 1
                self.C[i] = 0
            elif c == 2:
                # to right:
                self.P[i] += 1
                self.C[i] = 0
                
        # here, self.P[i] denotes the number of chopstics assigned for ith philos
        for i in range(N):
            if self.P[i] == 0:
                self.P[i] = random.randrange(2) # either think or hungry
            else:
                self.P[i] += 1 # one chopstick or eat

    def _get_vec(self):
        return tuple(self.P) + tuple(self.C)
                
    def _get_abs_vec(self):
        # abs_vec : < Think, Hungry, Single, Eat > e.g. < 1, 1, 1, 0 >
        return tuple(int(i in self.P) for i in range(4))
