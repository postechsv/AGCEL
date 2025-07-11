import random
#from AGCEL.common import Action

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
            
    def reset(self, to_state=None):
        if to_state == None:
            self.G_state = self.m.parseTerm(self.init())
        else:
            self.G_state = to_state
        self.state = self.obs(self.G_state)
        self.curr_rew = self.get_rew()
        # nbrs = (rhs,action) where rhs is the result of applying action on the current state
        self.nbrs = [(rhs, self.obs_act(label,sb)) for label in self.rules for rhs, sb, _, _ in self.G_state.apply(label)] # concrete
        return self.get_obs() 
    
    def get_obs(self):
        return {
            'G_state' : self.G_state, # ground state for Rewrite Theory : Maude.Term
            'state' : self.state, # observed state for MDP : Maude.Term
            'actions' : [a for s,a in self.nbrs] # List of (available) action objects
            # FIXME: actions may contain duplicates. better remove them later.
        }

    # action = Action obj = <rule label, abstract subs>
    def step(self, action):
        next_states = [s for s,a in self.nbrs if a == action] # TODO : s,a =/= action computed by self.nbrs are wasted (fix: only match in nbrs)
        if next_states == []:
            raise Exception("invalid action")
        reward = self.curr_rew
        obs = self.reset(random.choice(next_states))
        next_reward, done = self.curr_rew, self.is_done()
        return obs, next_reward, done

    # input: Maude.Term, output: Maude.Term
    def obs(self, term):
        term = self.m.parseTerm('obs(' + term.prettyPrint(0) + ')')
        term.reduce()
        return term

    # input: Maude.Substitution, output: dict
    def obs_act(self, label, subs):
        bindings = ' ; '.join([f"obs('{label},'{var.getVarName()},data({val.prettyPrint(0)}))" for var, val in subs])
        act = self.m.parseTerm("'" + label + " { " +  bindings + "}")
        act.reduce()
        return act

    def get_rew(self): # FIXME: actually, this should be get_utility
        t = self.m.parseTerm(f'reward({self.state.prettyPrint(0)})')
        t.reduce()
        return t.toFloat()

    def is_done(self):
        if self.nbrs == [] or self.curr_rew > 0.0000001: # TODO: or rew > 0 ?
            return True
        else:
            return False