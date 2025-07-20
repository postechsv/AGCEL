from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.neural_network import MLPRegressor
from AGCEL.Parser import parse_agcel_file
import itertools, os
from itertools import combinations

class RegressionScore:
    def __init__(self):
        self.model = None
        self.lookup = dict() 
        self.n_dim = None

    def train(self, qtable_file):
        data = parse_agcel_file(qtable_file)
        X = np.array([x for x, _ in data])
        y = np.array([y for _, y in data])

        self.n_dim = len(X[0])
        
        reg = LinearRegression()
        #reg = MLPRegressor(hidden_layer_sizes=(32,16), max_iter=1000, activation='relu')
        reg.fit(X, y)
        self.model = reg

        all_vecs = list(itertools.product([0,1], repeat=self.n_dim))
        preds = self.model.predict(all_vecs)

        self.make_lookup_table(qtable_file, all_vecs, preds)

        # Check Partial order preservation
        print('\n  --- SCORE COMPARISON (Ground Truth => Prediction) ---')
        preds = [self.score(vec) for vec in X]
        for i, (vec, gt, pred) in enumerate(zip(X, y, preds)):
            print(f"  {i+1:>3}: {vec} -> {gt:.4f} => Pred={pred:.4f}")

        order_total, order_match = 0, 0
        for i, j in combinations(range(len(X)), 2):
            real_order = y[i] > y[j]
            pred_order = preds[i] > preds[j]
            if y[i] != y[j]:
                order_total += 1
                if real_order == pred_order:
                    order_match += 1
        acc = order_match / order_total if order_total else 1.0
        print(f"\n[REGRESSION] Partial Order Preservation: {order_match}/{order_total} = {acc:.4f}")
    
    def make_lookup_table(self, qtable_file, all_vecs, preds):
        self.lookup = {}
        for vec, val in zip(all_vecs, preds):
            self.lookup[vec] = min(max(float(val), 0.0), 1.0)
        out_path = os.path.splitext(qtable_file)[0] + '.fullscores'
        with open(out_path, 'w') as f:
            for vec, score in zip(all_vecs, preds):
                f.write(f"{vec} |-> {score:.6f}\n")
        print(f"[REGRESSION] full score path : {out_path}")
        
    def score(self, vec):
        tvec = tuple(vec)
        return self.lookup.get(tvec, 0.0)

    def state_to_vec(self, state_term):
        feature_tuple = list(state_term.arguments())[0]
        features = list(feature_tuple.arguments())
        bool_vec = []
        for t in features:
            args = list(t.arguments())
            v = str(args[1].symbol())
            bool_val = 1 if v == "true" else 0
            bool_vec.append(bool_val)
        return bool_vec
    
    def get_value_function(self):
        return lambda state_term: self.score(self.state_to_vec(state_term))