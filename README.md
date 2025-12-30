# AGCEL: RL-based Heuristics Learning for Model Checking

Learn search heuristics using reinforcement learning (Q-Table and DQN) to guide state-space exploration in Maude-based model checking.


## Quick Start

```python
# Train
python3 train.py \
  benchmarks/dining-philosophers/dining-philosophers-focus-analysis.maude \
  init \
  deadlock \
  500 \
  trained/dining-philosophers-focus-init3-deadlock-500

# Test
python3 test.py \
  testcases/dining-philosophers-focus-5.maude \
  init \
  deadlock \
  trained/dining-philosophers-focus-init3-deadlock-500-c
```


## How It Works

1. **Training**: RL agents explore state space from a small initial state
2. **Learning**: Agents learn which states are closer to the goal
3. **Testing**: Learned heuristics guide search on larger instances


## Project Structure

```
.
├── README.md
├── LICENSE
│
├── AGCEL
│   ├── benchmarks
│   │   ├── bakery
│   │   │   ├── bakery.maude
│   │   │   ├── bakery-analysis.maude
│   │   │   └── bakery-focus-analysis.maude
│   │   ├── dining-philosophers
│   │   │   ├── dining-philosophers.maude
│   │   │   ├── dining-philosophers-analysis.maude
│   │   │   └── dining-philosophers-focus-analysis.maude
│   │   ├── filter
│   │   │   ├── filter.maude
│   │   │   ├── filter-analysis.maude
│   │   │   └── filter-focus-analysis.maude
│   │   ├── onethirdrule
│   │   │   ├── onethirdrule.maude
│   │   │   ├── onethirdrule-analysis.maude
│   │   │   └── onethirdrule-focus-analysis.maude
│   │   └── qlock
│   │       ├── qlock.maude
│   │       ├── qlock-analysis.maude
│   │       └── qlock-focus-analysis.maude
│   │
│   ├── AStar.py
│   ├── QLearning.py
│   ├── DQNLearning.py
│   ├── MaudeEnv.py
│   └── common.py
│
├── data_structures
│   ├── binary-tree.maude
│   ├── priority-queue.maude
│   └── test-list.maude
│
├── testcases
│   ├── bakery-*.maude
│   ├── bakery-focus-*.maude
│   ├── dining-philosophers-*.maude
│   ├── dining-philosophers-focus-*.maude
│   ├── filter-*.maude
│   ├── filter-focus-*.maude
│   ├── onethirdrule-*.maude
│   ├── onethirdrule-focus-*.maude
│   ├── qlock-*.maude
│   └── qlock-focus-*.maude
│
├── examples
├── traces
├── trained
│
├── tool.maude
├── tool2.maude
├── train.py
└── test.py
```


## Output Files

- `*-c.agcel` - Q-Table value function
- `*-c-d.pt` - DQN model checkpoint
- `*-c-v.json` - DQN vocabulary (predicate names)


## Key Metrics

- **n_states**: States explored before reaching goal
- **hit ratio**: Ratio of states encountered during search with known heuristic values

## Requirements

- Python
- PyTorch
- Maude