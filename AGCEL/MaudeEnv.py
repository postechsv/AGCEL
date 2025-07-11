import random
#from AGCEL.common import Action

"""
__init__ : Constructor for MaudeEnv
------------------------------------
- m (maude.Module) : Maude module to be analyzed
- goal (string) : goal property for the reachability analysis
- initializer (Unit -> string) : a function that returns the string for init term


reset : reset the current state to a particular state
------------------------------------

get_obs : returns a dictionary of observations to the current state
------------------------------------

step : given an action a, apply a to the current state and update it
------------------------------------

obs : given a maude term t, apply obs to t in the Maude level
------------------------------------

obs_act : given a maude term t for an action, return the python object for that action
------------------------------------

get_rew : get the utility for the current state
------------------------------------

is_done : returns true iff current state is an endstate
------------------------------------

"""

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