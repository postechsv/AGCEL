import random

class MaudeEnv():
    def __init__(self, m, goal, initializer):
        self.m = m # Maude module
        self.goal = m.parseTerm(goal) # Maude Term
        self.init = initializer # () -> String
        self.rules = []
        for rl in m.getRules():
            if not rl.getLabel() == None:
                self.rules.append(rl.getLabel())
        self.reset()
            
    def reset(self, init_state=None):
        if init_state == None:
            self.state = self.m.parseTerm(self.init())
        else:
            self.state = init_state
        self.actions = [rl for rl in self.rules if any(True for _ in self.state.apply(rl))]
        return self.get_obs() 
    
    def get_obs(self):
        return {
            'state' : self.state, # Maude.Term
            'astate' : self.abst(self.state), # Maude.Term
            'actions' : self.actions, # List of Strings
        }

    def step(self, action):
        next_states = [next_state for next_state, _, _, _ in self.state.apply(action)]
        if next_states == []:
            raise Exception("invalid action")
        obs = self.reset(random.choice(next_states))
        reward = 1.0 if self.is_goal() else 0.0
        done = True if reward == 1.0 else False # TODO: +done if no rewrite possible
        return obs, reward, done
    
    def abst(self, state):
        astate = self.m.parseTerm('abst(' + state.prettyPrint(0) + ')')
        astate.reduce()
        return astate
    
    def is_goal(self):
        t = self.m.parseTerm(f'{self.state.prettyPrint(0)} |= {self.goal.prettyPrint(0)}')
        t.reduce()
        return t.prettyPrint(0) == 'true'