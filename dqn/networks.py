import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
        Feed Forward Neural Net that approximates the policy
    """
    def __init__(self, state_size, action_size, seed, hidden_1_size=200, hidden_2_size=200):
        super(QNetwork, self).__init__()
        """
            ARGS:
            state_size(int):= dimension of each state
            action_size(int):= number of valid actions
            seed(int):= random seed
            hidden_1_size(int):= number of units in the first hidden layer
            hidden_2_size(int):= number of units in the second hidden layer
        """
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_1_size)
        self.fc2 = nn.Linear(hidden_1_size, hidden_2_size)
        self.fc3 = nn.Linear(hidden_2_size, action_size)
        
    def forward(self, state):
        """
            Builds a network that maps from the state space onto the action space
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out