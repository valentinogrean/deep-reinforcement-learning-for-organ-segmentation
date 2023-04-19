import numpy as np
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta

class ContinuousActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, learning_rate,
                 fc1_dims=256, fc2_dims=128, chkpt_dir='models/'):
        super(ContinuousActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir,
                                            'actor_continuous_ppo')
        
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1)
        self.fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(self.fc_input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.alpha = nn.Linear(fc2_dims, n_actions)
        self.beta = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.fc_input_dims)
        flat1 = F.relu(self.fc1(x))
        flat2 = F.relu(self.fc2(flat1))
        alpha = F.softplus(self.alpha(flat2))  + 1.0
        beta = F.softplus(self.beta(flat2))  + 1.0
        dist = Beta(alpha, beta)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1,256,256)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))


class ContinuousCriticNetwork(nn.Module):
    def __init__(self, input_dims, learning_rate,
                 fc1_dims=256, fc2_dims=128, chkpt_dir='models/'):
        super(ContinuousCriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir,
                                            'critic_continuous_ppo')
        
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1)
        self.fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(self.fc_input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.fc_input_dims)
        flat1 = F.relu(self.fc1(x))
        flat2 = F.relu(self.fc2(flat1))
    
        v = self.v(flat2)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, 256, 256)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))