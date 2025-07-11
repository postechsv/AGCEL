from AGCEL.MaudeEnv import MaudeEnv
import heapq

class Node():
    def __init__(self, m, t):
        self.m = m # Maude Module
        t.reduce()
        self.t = t # Maude Term of sort State
        #self.score = self.get_score(t)

    def __hash__(self):
        return hash(self.t)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.t == other.t # check if modulo ac
        return False

    def __lt__(self, other):
        return 0

    def get_score(self, V): # V: Value function (State -> Score)
        obs = self.m.parseTerm('obs(' + self.t.prettyPrint(0) + ')')
        obs.reduce()
        return V(obs)

    def get_next(self):
        #returns (next state, action) where action is applied to the current state to produce next state
        return [Node(self.m,t) for t, subs, path, nrew in self.t.search(1, self.m.parseTerm('X:State'), depth = 1)]

    def print_term(self):
        print(self.t.prettyPrint(0))

    def is_goal(self):
        t = self.m.parseTerm(f'{self.t.prettyPrint(0)} |= goal')
        t.reduce()
        #print(t.prettyPrint(0))
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
        p, d, n =  heapq.heappop(self.queue)
        return p, d, n

class Search():
    def search(self, init_node, V, bound):
        # arg: init term, Value dict, bound
        que = NodeQueue()
        vis = NodeSet()
        cnt = 0
        if init_node.is_goal(): return (True, init_node, cnt)
        que.push(init_node.get_score(V), 0, init_node)
        vis.add(init_node)
        while(True):
            cnt += 1
            if que.is_empty(): return (False, cnt)
            p, d, curr_node = que.pop()
            #print('i:', cnt, 'p:', p, 'd:', d)
            for next_node in curr_node.get_next():
                # goal check should be here due to value-shift w.r.t utility
                if next_node.is_goal(): return (True, next_node, cnt)
                if not vis.has(next_node):
                    que.push(next_node.get_score(V), d+1, next_node) # A*
                    #que.push(-(d+1), d+1, next_node) # bfs
                    vis.add(next_node)
        print('cnt:',cnt)