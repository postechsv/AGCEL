import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import datetime
from typing import Dict, Optional, Callable
from AGCEL.MaudeEnv import *

Experience = namedtuple('Experience', 
                        ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNLearner:
    def __init__(self, 
                 state_encoder: Callable,
                 input_dim: int,
                 num_actions: int,
                 learning_rate: float = 5e-4,
                 gamma: float = 0.9,
                 tau: float = 0.001,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.0002,
                 batch_size: int = 64,
                 buffer_size: int = 10000,
                 update_frequency: int = 4,
                 target_update_frequency: int = 500,
                 device: Optional[str] = None):
        
        self.state_encoder = state_encoder
        self.input_dim = input_dim
        self.num_actions = num_actions
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not torch.cuda.is_available():
                print("cuda is not available: cpu")
        else:
            self.device = torch.device(device)
        
        self.q_network = DQN(input_dim, num_actions).to(self.device)
        self.target_network = DQN(input_dim, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.tau = tau
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        self.training_step = 0
        self.episode_count = 0
        self.loss_history = []
        
        self.value_cache = {}
    
    def select_action(self, env, obs: Dict, epsilon: Optional[float] = None) -> Optional[int]:
        if epsilon is None:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.epsilon_decay * self.episode_count)
        
        state = obs['state']
        g_state = obs.get('G_state')
        if g_state is None:
            raise KeyError("'G_state' not found in observation")
        
        action_mask = env.action_mask(state=g_state)
        
        if len(action_mask) != self.num_actions:
            raise ValueError(f"Action mask length {len(action_mask)} != num_actions {self.num_actions}")
        
        legal_actions = [i for i, valid in enumerate(action_mask) if valid]
        
        if not legal_actions:
            return None
        
        if random.random() < epsilon:
            return random.choice(legal_actions)
        else:
            with torch.no_grad():
                state_tensor = self.state_encoder(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).squeeze(0)
                
                masked_q_values = q_values.clone()
                for i in range(self.num_actions):
                    if not action_mask[i]:
                        masked_q_values[i] = -float('inf')
                
                return masked_q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        state_tensor = self.state_encoder(state)
        next_state_tensor = self.state_encoder(next_state) if next_state is not None else torch.zeros_like(state_tensor)
        
        self.replay_buffer.push(
            state_tensor,
            action,
            reward,
            next_state_tensor,
            done
        )
    
    def optimize_model(self, env=None):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        
        state_batch = torch.stack([e.state for e in batch]).to(self.device)
        action_batch = torch.tensor([e.action for e in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=self.device)
        next_state_batch = torch.stack([e.next_state for e in batch]).to(self.device)
        done_batch = torch.tensor([e.done for e in batch], dtype=torch.float32, device=self.device)
        
        current_q_values = self.q_network(state_batch).gather(1, action_batch).squeeze(1)
        
        with torch.no_grad():
            next_q_values_online = self.q_network(next_state_batch)
            next_actions = next_q_values_online.argmax(dim=1, keepdim=True)

            next_q_values_target = self.target_network(next_state_batch)
            next_q_values = next_q_values_target.gather(1, next_actions).squeeze(1)

            target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        
    def update_target_network(self):
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def train(self, env, n_episodes: int, max_steps: int = 10000):
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            self.episode_count = episode
            obs = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                action_idx = self.select_action(env, obs)
                
                if action_idx is None:
                    break
                
                next_obs, reward, done = env.step_indexed(action_idx)
                episode_reward += reward
                
                self.store_experience(
                    obs['state'],
                    action_idx,
                    reward,
                    next_obs['state'] if not done else None,
                    done
                )
                
                if self.training_step % self.update_frequency == 0:
                    self.optimize_model()
                
                if self.training_step % self.target_update_frequency == 0:
                    self.update_target_network()
                
                self.training_step += 1
                obs = next_obs
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step + 1)
            
            # if episode % 100 == 0:
            #     avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            #     avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
            #     print(f"    Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}, ")
        
        print("Training completed!")
        
        return episode_rewards, episode_lengths
    
    def get_value_function(self) -> Callable:
        self.q_network.eval()
        
        @torch.no_grad()
        def V(obs_term, g_state=None) -> float:
            if obs_term is not None:
                obs_str = obs_term.prettyPrint(0)
                if obs_str in self.value_cache:
                    return self.value_cache[obs_str]
            
            state_tensor = self.state_encoder(obs_term).unsqueeze(0).to(self.device)
            
            q_values = self.q_network(state_tensor).squeeze(0)
            max_q = q_values.max().item()
            
            if obs_term is not None:
                self.value_cache[obs_str] = max_q
            
            return max_q
        
        V.needs_obs = True
        return V
    
    def save(self, path: str):
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'loss_history': self.loss_history,
            'hyperparameters': {
                'input_dim': self.input_dim,
                'num_actions': self.num_actions,
                'gamma': self.gamma,
                'tau': self.tau,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size
            }
        }
        torch.save(checkpoint, path)
        #print(f"Model saved to {path}")
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        self.loss_history = checkpoint['loss_history']
        
        if 'hyperparameters' in checkpoint:
            hp = checkpoint['hyperparameters']
            self.gamma = hp.get('gamma', self.gamma)
            self.tau = hp.get('tau', self.tau)
            self.batch_size = hp.get('batch_size', self.batch_size)
        
        print(f"Model loaded from {path}")
    
    def dump_value_function(self, filename: str, module_name: str, goal: str):
        with open(filename, 'w') as f:
            f.write(f'--- DQN-learned heuristic generated at {datetime.datetime.now()}\n')
            f.write('mod DQN-HEURISTIC is\n')
            f.write(f'  pr HEURISTIC-BASE . pr {module_name} .\n')
            f.write('  var S : MDPState . var A : MDPAct .\n')
            
            for state_str, value in self.value_cache.items():
                f.write(f'  eq score({goal}, {state_str}) = {value} .\n')
            
            f.write(f'  eq score({goal}, S) = 0.0 [owise] .\n')
            f.write('endm\n')
        
        print(f"Value function exported to {filename}")