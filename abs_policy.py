import numpy as np
import numpy.ma as ma
import random

def abs_greedy_policy(Qtable, state):
    # Exploitation: take the action with the highest state, action value
    vec = state["abs_vec"]
    mask = state["abs_mask"]
    mask = np.where(True, mask^1, mask) # flip 0 & 1
    if 0 in mask:
        masked_Q = ma.masked_array(Qtable[vec][:], mask=mask)
        action = np.unravel_index(np.argmax(masked_Q), (2,2,2,2))
    else:
        action = -1 # deadlock
    return action

def abs_epsilon_greedy_policy(Qtable, state, epsilon):
    # Randomly generate a number between 0 and 1
    random_num = random.uniform(0, 1)
    # if random_num > greater than epsilon --> exploitation
    if random_num > epsilon:
        # Take the action with the highest value given a state
        # np.argmax can be useful here
        action = abs_greedy_policy(Qtable, state)
    # else --> exploration
    else:
        mask = state["abs_mask"]
        if 1 in mask:
            #action = env.abs_action_space.sample(mask=mask) # TODO : masked sample
            action = tuple(random.choice(np.transpose(np.nonzero(mask))))
        else:
            action = -1
    return action
