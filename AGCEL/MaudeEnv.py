import random

class MaudeEnv():
    def __init__(self, m, goal, initializer):
        self.m = m
        self.goal = m.parseTerm(goal)
        self.init = initializer
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
        self.nbrs = [
            (rhs, self.obs_act(label,sb)) 
            for label in self.rules 
            for rhs, sb, _, _ in self.G_state.apply(label)
        ]
        return self.get_obs()

    def step(self, action):
        next_states = [s for s,a in self.nbrs if a.equal(action)]
        if  next_states == []:
            raise Exception("invalid action")
        obs = self.reset(random.choice(next_states))
        return obs, self.curr_reward, self.is_done()
    
    def get_obs(self):
        acts = [a for _,a in self.nbrs]
        out = []
        for t in acts:  # t = self.obs_act(rule label, substitution)
            if not any(t.equal(u) for u in out):
                out.append(t)
        return {
            'G_state' : self.G_state,
            'state' : self.state,
            'actions' : out
        }

    def is_done(self):
        t = self.m.parseTerm(f'{self.G_state.prettyPrint(0)} |= {self.goal.prettyPrint(0)}')
        t.reduce()
        return (t.prettyPrint(0) == 'true') or self.nbrs == [] or self.curr_reward > 1e-7
    
    def get_reward(self):
        t = self.m.parseTerm(f'reward({self.state.prettyPrint(0)})')
        t.reduce()
        return t.toFloat()

    def action_mask(self, state=None):
        term = state if state is not None else self.G_state
        mask = []
        for label in self.rules:
            has_app = any(term.apply(label))
            mask.append(1 if has_app else 0)
        return mask
    
    def step_indexed(self, action_idx):
        if action_idx >= len(self.rules):
            raise ValueError(f"Invalid action index: {action_idx}")
        
        label = self.rules[action_idx]
        next_states = [s for s, a in self.nbrs if str(a).startswith(f"'{label}")]
        
        if not next_states:
            next_states = [rhs for rhs, _, _, _ in self.G_state.apply(label)]
            if not next_states:
                raise ValueError(f"Action {label} not applicable")
            
        obs = self.reset(random.choice(next_states))
        reward = self.curr_reward
        
        return obs, reward, self.is_done()

    def obs(self, term):
        term = self.m.parseTerm('obs(' + term.prettyPrint(0) + ')')
        term.reduce()
        return term

    def obs_act(self, label, subs):
        bindings = ' ; '.join([f"obs('{label},'{var.getVarName()},data({val.prettyPrint(0)}))" for var, val in subs])
        act = self.m.parseTerm("'" + label + " { " +  bindings + "}")
        act.reduce()
        return act