import random

import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class DQNLearner():
    def __init__(self, env, state_encoder, input_dim, num_actions,
                 gamma, lr, tau):
        self.env = env
        self.encoder = state_encoder

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net      = DQN(input_dim, num_actions).to(self.device)
        self.target_net = DQN(input_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=lr)

    def select_action(self, obs, epsilon):
        s_term = obs['state']
        x = self.encoder(s_term).unsqueeze(0).to(self.device)
        q_values = self.q_net(x)[0] # q values for all action

        mask = torch.tensor(self.env.action_mask(state=s_term), dtype=torch.bool, device=self.device)
        q_values[~mask] = -1e9  # low q value for illegal actions

        # eps_greedy_policy
        if random.random() > epsilon:   # exploitation
            return int(torch.argmax(q_values).item())   # choose action with max q value
        else:                           # exploration
            legal = torch.nonzero(mask, as_tuple=False).view(-1).tolist()
            return random.choice(legal) # choose random legal action
        
    def train(self, n_training_episodes):
        for episode in range(n_training_episodes):
            epsilon = 0

            obs = self.env.reset()
            done = False
            
            while not done:
                a_idx = self.select_action(obs, epsilon)
                next_terms = self.env.step_by_index(a_idx)

                if not next_terms:
                    next_q = 0.0    # if no next state, Q=0
                else:
                    with torch.no_grad():
                        next_qs = []
                        for nt in next_terms:   # for each (legal) next state
                            nt_tensor = self.encoder(nt).unsqueeze(0).to(self.device)
                            q = self.target_net(nt_tensor)[0]   # Q for next states from target net
                            mask = torch.tensor(self.env.action_mask(state=nt), dtype=torch.bool, device=self.device)
                            q[~mask] = -1e9     # mask illegal actions
                            next_qs.append(torch.max(q).item()) # best Q for this next state
                        next_q = sum(next_qs) / len(next_qs)    # mean Q over next states