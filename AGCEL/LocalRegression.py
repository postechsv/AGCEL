from AGCEL.Parser import parse_agcel_file
from sklearn.linear_model import LinearRegression
import itertools
import numpy as np
import os

def hamming(v1, v2):
    return sum(a != b for a, b in zip(v1, v2))

class LocalRegressionScore:
    def __init__(self, k=1, min_nbrs=3):
        self.k = k
        self.min_nbrs = min_nbrs
        self.table = {}
        self.n_dim = None
        self.cache = {}

    def train(self, agcel_file):
        data = parse_agcel_file(agcel_file)
        self.table = {tuple(vec): val for vec, val in data}
        self.n_dim = len(data[0][0])
        self.make_lookup_table(agcel_file)

    def predict(self, vec):
        vec = tuple(vec)
        if vec in self.cache:
            return self.cache[vec]
        nbrs = []
        for mask in itertools.product([0, 1], repeat=self.n_dim):
            if hamming(vec, mask) <= self.k and mask in self.table:
                nbrs.append(mask)
        if len(nbrs) >= self.min_nbrs:
            X = np.array(nbrs)
            y = np.array([self.table[v] for v in nbrs])
            model = LinearRegression()
            model.fit(X, y)
            pred = float(model.predict([vec])[0])
        else:
            pred = 0.0
        pred = min(max(pred, 0.0), 1.0)
        self.cache[vec] = pred
        return pred

    def state_to_vec(self, state_term):
        feature_tuple = list(state_term.arguments())[0]
        return [1 if str(list(t.arguments())[1].symbol()) == "true" else 0
                for t in list(feature_tuple.arguments())]

    def make_lookup_table(self, agcel_file):
        all_vecs = list(itertools.product([0, 1], repeat=self.n_dim))
        out_path = os.path.splitext(agcel_file)[0] + '.lr'
        with open(out_path, 'w') as f:
            for vec in all_vecs:
                score = self.predict(vec)
                f.write(f"{vec} |-> {score:.6f}\n")
        print(f"[LOCAL REG] full score path : {out_path}")

    def get_value_function(self):
        return lambda state_term: self.predict(self.state_to_vec(state_term))