import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # TODO
    def forward(self, x):
        pass  # TODO

class DQNLearner():
    def __init__(self, env, state_encoder, input_dim, num_actions,
                 gamma, lr, tau):
        self.env = env
        self.encoder = state_encoder