from collections import deque, namedtuple
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_states', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

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

        self.gamma = gamma
        self.tau = tau

        self.replay = ReplayBuffer(capacity=10000)
        self.batch_size = 64

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
        
    def compute_target_q(self, batch):  # mean-max over next states
        target_qs = []
        for _, _, reward, next_states, done in batch:
            if done or not next_states:
                target_qs.append(torch.tensor(reward, device=self.device))
            else:
                qmaxes = []
                for nt in next_states:   # for each (legal) next state
                    q = self.target_net(nt.unsqueeze(0))[0]   # Q for next states from target net
                    mask = torch.tensor(self.env.action_mask(state=None), dtype=torch.bool, device=self.device)
                    q[~mask] = -1e9     # mask illegal actions
                    qmaxes.append(torch.max(q).item())  # best Q for next state
                q_mean = sum(qmaxes) / len(qmaxes)      # mean Q over next states
                target_qs.append(torch.tensor(reward + self.gamma * q_mean, device=self.device))
        return torch.stack(target_qs)

    def optimize_model(self):
        if len(self.replay) < self.batch_size:
            return  # not enough samples to train

        batch = self.replay.sample(self.batch_size)
        batch = Transition(*zip(*batch))    # unpack transitions

        state_batch  = torch.stack(batch.state)     # [B, D] states
        action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(self.device)    # [B,1] actions
        pred_q       = self.q_net(state_batch).gather(1, action_batch).squeeze(1)   # [B] predicted Q(s,a)

        target_q     = self.compute_target_q(batch) # [B]
        loss         = nn.functional.smooth_l1_loss(pred_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()         # backpropagation
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()   # update network

    def train(self, n_training_episodes):
        max_steps = 300
        max_epsilon = 1.0
        min_epsilon = 0.05
        decay_rate = 0.0005

        for episode in tqdm(range(n_training_episodes)):
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

            obs = self.env.reset()
            done = False
            
            for _ in range(max_steps):
                s_term = obs['state']
                s_tensor = self.encoder(s_term).to(self.device)
                a_idx = self.select_action(obs, epsilon)

                next_terms = self.env.step_by_index(a_idx)

                if not next_terms:
                    next_q = 0.0    # if no next state, Q=0
                else:
                    with torch.no_grad():
                        next_qs = []
                        for nt in next_terms:
                            nt_tensor = self.encoder(nt).unsqueeze(0).to(self.device)
                            q = self.target_net(nt_tensor)[0]
                            mask = torch.tensor(self.env.action_mask(state=nt), dtype=torch.bool, device=self.device)
                            q[~mask] = -1e9
                            next_qs.append(torch.max(q).item())
                        next_q = sum(next_qs) / len(next_qs)

                reward = self.env.curr_reward
                done = self.env.is_done()       # check reward, termination from env

                self.replay.push(s_tensor, a_idx, reward, [self.encoder(s).to(self.device) for s in next_terms], done)

                # target = reward + self.gamma * next_q                   # target Q
                # pred_q = self.q_net(s_tensor.unsqueeze(0))[0][a_idx]    # predicted Q

                # loss = nn.functional.smooth_l1_loss(pred_q, torch.tensor(target, device=self.device))   # smooth l1 loss
                # self.optimizer.zero_grad()
                # loss.backward()         # backpropagation
                # torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)    # gradient clipping
                # self.optimizer.step()   # update weights

                self.optimize_model()

                self.soft_update()            # update target net
                obs = self.env.reset(random.choice(next_terms)) if next_terms else obs  # move to next state

                if done:
                    break

    def soft_update(self):  # update target net
        for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data) # tau=1 -> hard 
            
    def make_v_dict(self):  # V(s)=max_a Q(s,a)
        self.v_dict = dict()
        for s in self.q_dict_keys():
            x = self.encoder(s).unsqueeze(0).to(self.device)
            q = self.q_net(x)[0]
            mask = torch.tensor(self.env.action_mask(state=s), dtype=torch.bool, device=self.device)
            q[~mask] = -1e9
            self.v_dict[s] = float(torch.max(q).item())

    def dump_value_function(self, filename):
        with open(filename, 'w') as f:
            for s, v in self.v_dict.items():
                f.write(f'{s} |-> {v}\n')

    def load_value_function(self, filename, m):
        self.v_dict = dict()
        with open(filename, 'r') as f:
            for line in f:
                state, value = line.split(" |-> ")
                state = m.parseTerm(state)
                state.reduce()
                self.v_dict[state] = float(value)