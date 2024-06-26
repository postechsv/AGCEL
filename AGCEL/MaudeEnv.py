import random
from AGCEL.common import Action

class MaudeEnv():
    def __init__(self, m, goal, initializer, abst_mode='full'):
        '''
        abst_mode : {label, full}
        '''
        self.m = m # Maude module
        self.goal = m.parseTerm(goal) # Maude Term
        self.init = initializer # () -> String
        self.abst_mode = abst_mode
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
        # nbrs = (rhs,action) where rhs is the result of applying action on the current state
        self.nbrs = [(rhs, Action(label, self.abst_subs(label,sb))) for label in self.rules for rhs, sb, _, _ in self.state.apply(label)] # concrete
        #self.nbrs = [(rhs, Action(label, self.abst_subs2(rhs))) for label in self.rules for rhs, sb, _, _ in self.state.apply(label)] # abstract
        return self.get_obs() 
    
    def get_obs(self):
        return {
            'state' : self.state, # Maude.Term
            #'astate' : self.abst(self.state), # Maude.Term
            'astate' : self.heur(self.state), # Maude.Term
            'actions' : [a for s,a in self.nbrs] # List of (available) action objects
        }

    # action = Action obj = <rule label, abstract subs>
    def step(self, action):
        next_states = [s for s,a in self.nbrs if a == action]
        if next_states == []:
            raise Exception("invalid action")
        obs = self.reset(random.choice(next_states))
        reward, done = 0.0, False
        if self.nbrs == []:
            done = True
        if self.is_goal():
            reward = 1.0
            done = True
        return obs, reward, done
    
    # input: Maude.Term, output: Maude.Term
    def abst(self, term):
        term = self.m.parseTerm('abst(' + term.prettyPrint(0) + ')')
        term.reduce()
        return term

    # input: Maude.Term, output: Maude.Term
    def heur(self, term):
        term = self.m.parseTerm('heur(' + term.prettyPrint(0) + ')')
        term.reduce()
        return term

    # input: Maude.Substitution, output: dict
    def abst_subs(self, label, subs):
        if self.abst_mode == 'label':
            return None
        asubs = dict()
        for var, val in subs:
            val = self.m.parseTerm(f"abst('{label},'{var.getVarName()},data({val.prettyPrint(0)}))")
            val.reduce()
            asubs[var] = val
        return asubs

    # TEST
    def abst_subs2(self, rhs):
        term = self.m.parseTerm(f'getASubs({rhs.prettyPrint(0)})')
        term.reduce()
        return term

    def is_goal(self):
        t = self.m.parseTerm(f'{self.state.prettyPrint(0)} |= {self.goal.prettyPrint(0)}')
        t.reduce()
        return t.prettyPrint(0) == 'true'
