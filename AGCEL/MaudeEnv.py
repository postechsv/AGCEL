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
            
    def reset(self, to_state=None):
        if to_state == None:
            self.G_state = self.m.parseTerm(self.init())
        else:
            self.G_state = to_state
        self.state = self.obs(self.G_state)
        self.curr_reward = self.get_reward()
        # nbrs = (rhs,action) where rhs is the result of applying action on the current state
        self.nbrs = [(rhs, self.obs_act(label,sb)) for label in self.rules for rhs, sb, _, _ in self.G_state.apply(label)] # concrete
        return self.get_obs()

    # action = Action obj = <rule label, abstract subs>
    def step(self, action):
        next_states = [s for s,a in self.nbrs if a.equal(action)] # TODO : s,a =/= action computed by self.nbrs are wasted (fix: only match in nbrs)
        if next_states == []:
            raise Exception("invalid action")
        obs = self.reset(random.choice(next_states))
        return obs, self.curr_reward, self.is_done()
    
    def get_obs(self):
        acts = [a for _,a in self.nbrs] # List of (available) action objects
        out = []
        for t in acts:
            if not any(t.equal(u) for u in out):
                out.append(t)
        return {
            'G_state' : self.G_state,   # ground state for Rewrite Theory : Maude.Term
            'state' : self.state,       # observed state for MDP : Maude.Term
            'actions' : out
        }

    def is_done(self):
        if self.G_state.equal(self.goal):
            return True
        return self.nbrs == [] or self.curr_reward > 1e-7

    def get_reward(self): # FIXME: actually, this should be get_utility
        t = self.m.parseTerm(f'reward({self.state.prettyPrint(0)})')
        t.reduce()
        return t.toFloat()

    def action_mask(self):
        pass

    def step_by_index(self):
        pass

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