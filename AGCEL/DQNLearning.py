from collections import deque, namedtuple
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_obs', 'next_term', 'done'))

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
        s_obs  = obs['state']
        s_term = obs['G_state']
        x = self.encoder(s_obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_net(x)[0] # q values for all action

        mask = torch.tensor(self.env.action_mask(state=s_term), dtype=torch.bool, device=self.device)

        q_values[~mask] = -1e9  # low q value for illegal actions

        legal = torch.nonzero(mask, as_tuple=False).view(-1).tolist()
        if not legal:   # empty legal actions
            return None
        
        # eps_greedy_policy
        if random.random() > epsilon:   # exploitation
            return int(torch.argmax(q_values).item())   # choose action with max q value
        else:                           # exploration
            return random.choice(legal) # choose random legal action
        
    def compute_target_q(self, batch):
        targets = []
        with torch.no_grad():
            for r, next_obs, next_term, done in zip(batch.reward, batch.next_obs, batch.next_term, batch.done):
                r_t = torch.tensor(r, dtype=torch.float32, device=self.device)
                if done:
                    targets.append(r_t)
                else:
                    q_next = self.target_net(next_obs.unsqueeze(0).to(self.device))[0]
                    mask = torch.tensor(self.env.action_mask(state=next_term), dtype=torch.bool, device=self.device)
                    q_next[~mask] = -1e9
                    targets.append(r_t + self.gamma * torch.max(q_next))
        return torch.stack(targets)

    def optimize_model(self):
        if len(self.replay) < self.batch_size:
            return  # not enough samples to train

        batch = self.replay.sample(self.batch_size)
        batch = Transition(*zip(*batch))    # unpack transitions

        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)    # [B,1]
        state_batch  = torch.stack(batch.state).to(self.device)     # [B, D]
        pred_q       = self.q_net(state_batch).gather(1, action_batch).squeeze(1)
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
                s_tensor = self.encoder(obs['state']).to(self.device)
                a_idx = self.select_action(obs, epsilon)

                if a_idx is None:
                    break

                obs2, reward, done = self.env.step_indexed(a_idx)
                next_tensor = self.encoder(obs2['state']).to(self.device)
                self.replay.push(s_tensor, a_idx, reward, next_tensor, obs2['G_state'], done)

                self.optimize_model()
                self.soft_update()

                obs = obs2

                if done:
                    break

    def soft_update(self):  # update target net
        for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data) # tau=1 -> hard 

    # def get_value_function(self):
    #     def V(obs_term, g_state=None):
    #         with torch.no_grad():
    #             x = self.encoder(obs_term).unsqueeze(0).to(self.device)
    #             q = self.q_net(x)[0]
    #             if g_state is not None:
    #                 mask = torch.tensor(self.env.action_mask(state=g_state), dtype=torch.bool, device=self.device)
    #                 q[~mask] = -1e9
    #         return float(torch.max(q).item())
    #     return V

    def get_value_function(self):
        self.q_net.eval()
        enc_cache  = {}
        mask_cache = {}

        def V(obs_term, g_state=None):
            obs_str = obs_term.prettyPrint(0)
            x = enc_cache.get(obs_str)
            if x is None:
                x = self.encoder(obs_term).unsqueeze(0).to(self.device)
                enc_cache[obs_str] = x

            with torch.no_grad():
                q = self.q_net(x)[0]

            if g_state is not None:
                g_str = g_state.prettyPrint(0)
                mask = mask_cache.get(g_str)
                if mask is None:
                    mlist = self.env.action_mask(state=g_state)
                    mask = torch.tensor(mlist, dtype=torch.bool, device=self.device)
                    mask_cache[g_str] = mask
                q = q.masked_fill(~mask, -1e9)

            return float(q.max().item())
        return V

    def save_model(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path):
        sd = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(sd)
        self.target_net.load_state_dict(sd)