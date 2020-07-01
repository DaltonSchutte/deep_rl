import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_init(layer):
    fan_in = layer.weight.data.size()[0]
    bound = 1.0 / np.sqrt(fan_in)
    return (-bound, bound)

class Actor(nn.Module):
    def __init__(self, state_size, action_dim, fc1_width, fc2_width, seed):
        super().__init__()
        self.state_size = state_size
        self.action_dim = action_dim
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, fc1_width)
        self.fc2 = nn.Linear(fc1_width, fc2_width)
        self.fc3 = nn.Linear(fc2_width, action_dim)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*weight_init(self.fc1))
        self.fc2.weight.data.uniform_(*weight_init(self.fc2))
        self.fc3.weight.data.uniform_(-0.003, 0.003)
        
    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = F.selu(self.fc1(state))
        x = F.selu(self.fc2(x))
        out = F.tanh(self.fc3(x))
        
        return out
        
        
class Critic(nn.Module):
    def __init__(self, state_size, action_dim, fc1_width, fc2_width, seed):
        super().__init__()
        self.state_size = state_size
        self.action_dim = action_dim
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, fc1_width)
        self.fc2 = nn.Linear(fc1_width+action_dim, fc2_width)
        self.fc3 = nn.Linear(fc2_width, 1)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*weight_init(self.fc1))
        self.fc2.weight.data.uniform_(*weight_init(self.fc2))
        self.fc3.weight.data.uniform_(-0.003, 0.003)
    
    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = F.selu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = F.selu(self.fc2(x))
        out = self.fc3(x)
        
        return out
    