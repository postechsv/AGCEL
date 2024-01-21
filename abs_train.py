from tqdm import tqdm
import numpy as np
from abs_policy import *

# TODO: abstract these aguments with config
def abs_train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, learning_rate, gamma, env, max_steps, Qtable):
    for episode in tqdm(range(n_training_episodes)):
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        # Reset the environment
        state, info = env.reset()
        step = 0
        done = False

        # repeat
        for step in range(max_steps):
            #print('===')
            #print(env._get_obs())
            # Choose the action At using epsilon greedy policy
            action = abs_epsilon_greedy_policy(Qtable, state, epsilon)
            # assert action not -1
            if action == -1:
                break
            
            #print(action)

            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done, info = env.abs_step(action)
            
            #print(done)
            
            vec = state["abs_vec"]
            new_vec = new_state["abs_vec"]

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            Qtable[vec][action] = Qtable[vec][action] + learning_rate * (
                reward + gamma * np.max(Qtable[new_vec]) - Qtable[vec][action]
            )

            # If terminated or truncated finish the episode
            if done:
                break

            # Our next state is the new state
            state = new_state
    print('training done!')
    return Qtable
