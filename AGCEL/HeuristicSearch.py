from AGCEL.MaudeEnv import MaudeEnv
import heapq

class TermWrapper():
    def __init__(self,t):
        self.t = t
        
    def __lt__(self, other):
        return 0

class HeuristicSearch(MaudeEnv):
    def __init__(self, m, goal, initializer, qt):
        MaudeEnv.__init__(self, m, goal, initializer)
        self.qt = qt
        self.last_init = self.state
        
    def get_nbrs(self):
        #returns (next state, action) where action is applied to the current state to produce next state
        return [(t, path()[1].getLabel()) for t, subs, path, nrew in self.state.search(1, self.m.parseTerm('X:State'), depth = 1)]
    
    def score(self, s, a):
        astate = self.abst(s)
        return self.qt.get_q(astate, a)
        
    def search(self, mode='qhs', max_step=1000):
        visited = set()
        i = 0
        queue = [(i,TermWrapper(self.state))] # (priority, concrete_state)

        while not queue == [] and i < max_step:
            state = heapq.heappop(queue)[1].t
            obs = self.reset(state)
            #print(t)
            if state in visited:
                continue
            i += 1
            visited.add(state)
            s = obs['astate']
            if self.is_goal():
                print('goal reached!')
                #print('t:', t)
                #print('num steps:', i)
                break
            # nbrs = [(v, av) for (a, v, av) in env.next_actions if not v in visited] # unvisited next vecs
            if mode == 'bfs': # bfs
                q_items = [(i, TermWrapper(next_state)) for (next_state, a) in self.get_nbrs()]
            elif mode == 'qhs': # qhs
                q_items = [(-self.score(state, a), TermWrapper(next_state)) for (next_state, a) in self.get_nbrs()] # prioritized nbrs
            for item in q_items:
                heapq.heappush(queue, item) # queue,item
        return i