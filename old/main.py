from dp_env import DiningPhilosEnv
import numpy as np
from abs_policy import *
from abs_train import *

env = DiningPhilosEnv(5)

def init_q_table(state_space, action_space):
  Qtable = np.zeros(state_space + action_space)
  return Qtable

absQtable = init_q_table((2,2,2,2), (2,2,2,2))

# Training parameters
n_training_episodes = 100  # Total training episodes
learning_rate = 0.7  # Learning rate

# Evaluation parameters
n_eval_episodes = 100  # Total number of test episodes

# Environment parameters
#env_id = "FrozenLake-v1"  # Name of the environment
max_steps = 99  # Max steps per episode
gamma = 0.95  # Discounting rate
eval_seed = []  # The evaluation seed of the environment

# Exploration parameters
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.0005  # Exponential decay rate for exploration prob

absQtable = abs_train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, learning_rate, gamma, env, max_steps, absQtable)
np.save('trained_abs_qtable.npy', absQtable)
