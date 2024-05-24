from AGCEL.MaudeEnv import MaudeEnv
import heapq

class TermWrapper():
    def __init__(self,t):
        self.t = t
        
    def __lt__(self, other):
        return 0

class HeuristicSearch(MaudeEnv):
    def __init__(self, m, goal, initializer, qt, abst_mode='full'):
        MaudeEnv.__init__(self, m, goal, initializer, abst_mode=abst_mode)
        self.qt = qt
        self.last_init = self.state
    
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
                break
            if mode == 'bfs':
                q_items = [(i, TermWrapper(next_state)) for (next_state, a) in self.nbrs]
            elif mode == 'dfs':
                q_items = [(-i, TermWrapper(next_state)) for (next_state, a) in self.nbrs]
            elif mode == 'qhs':
                q_items = [(-self.score(state, a), TermWrapper(next_state)) for (next_state, a) in self.nbrs] # prioritized nbrs
            elif mode == 'const':
                q_items = [(0, TermWrapper(next_state)) for (next_state, a) in self.nbrs] # prioritized nbrs
            for item in q_items:
                heapq.heappush(queue, item) # queue,item
        return i