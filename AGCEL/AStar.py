import heapq

class Node():
    def __init__(self, m, t):
        self.m = m # Maude Module
        t.reduce()
        self.t = t # Maude Term of sort State
        self.key = None
        self._obs = None

    def _ensure_key(self):
        if self.key is None:
            self.key = self.t.prettyPrint(0)
    
    def __hash__(self):
        return hash(self.t)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.t == other.t # check if modulo ac
        return False

    def __lt__(self, other):
        return 0

    def get_obs(self):
        obs = self.m.parseTerm('obs(' + self.t.prettyPrint(0) + ')')
        obs.reduce()
        return obs

    def get_score(self, V): # V: Value function (State -> Score)
        # obs = self.m.parseTerm('obs(' + self.t.prettyPrint(0) + ')')
        # obs.reduce()
        # return V(obs, self.t)
        if getattr(V, 'needs_obs', True) is False:
            return V(None, self.t)
        if self._obs is None:
            self._ensure_key()
            self._obs = self.m.parseTerm('obs(' + self.key + ')')
            self._obs.reduce()
        return V(self._obs, self.t)

    def get_next(self):
        #returns (next state, action) where action is applied to the current state to produce next state
        return [Node(self.m,t) for t, subs, path, nrew in self.t.search(1, self.m.parseTerm('X:State'), depth = 1)]

    def print_term(self):
        print(self.t.prettyPrint(0))

    def is_goal(self):
        # t = self.m.parseTerm(f'{self.t.prettyPrint(0)} |= goal')
        self._ensure_key()
        t = self.m.parseTerm(f'{self.key} |= goal')
        t.reduce()
        return t.prettyPrint(0) == 'true'

class NodeSet():
    def __init__(self):
        self.set = set()

    def add(self, node):
        self.set.add(node)

    def remove(self, node):
        self.set.remove(node)

    def has(self, node):
        return node in self.set

class NodeQueue():
    def __init__(self):
        self.queue = []

    def is_empty(self):
        return self.queue == [] 

    def push(self, score, depth, node):
        #self.queue = [node] + self.queue
        heapq.heappush(self.queue, (-score, depth, node))

    def pop(self):
        #return 0, self.queue.pop()
        p, d, n =  heapq.heappop(self.queue)    # priority(min score), depth, node
        return p, d, n

class Search():
    def search(self, init_node, V, bound):
        que = NodeQueue()
        vis = NodeSet()
        cnt = 0
        hit_cnt = 0
        state_cnt = 0

        if init_node.is_goal(): return (True, init_node, 0)
        que.push(init_node.get_score(V), 0, init_node)
        vis.add(init_node)

        while True:
            cnt += 1
            if que.is_empty(): return (False, cnt)
            _, d, curr_node = que.pop()

            for next_node in curr_node.get_next():
                if vis.has(next_node): continue
                vis.add(next_node)
                score = next_node.get_score(V)
                state_cnt += 1
                if abs(score) > 1e-8:
                    hit_cnt += 1
                if next_node.is_goal():
                    if hit_cnt == 0:
                        print(f'[SEARCH] lookup cnt: {state_cnt}')
                    else:
                        print(f'[SEARCH] hit ratio: {hit_cnt}/{state_cnt} = {hit_cnt/state_cnt:.4f}')
                    return (True, next_node, cnt)
                que.push(score, d + 1, next_node)