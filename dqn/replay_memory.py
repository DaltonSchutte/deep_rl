import torch
import random
import numpy as np
from collections import namedtuple, deque

class ReplayBuffer:
    """
        Buffer to store experience tuples
    """
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
            ARGS:
            action_size(int):= number of valid actions
            buffer_size(int):= max number of experiences the buffer can store
            batch_size(int):= number of experiences to sample from the buffer
            seed(int):= random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names=['state',
                                                                'action',
                                                                'reward',
                                                                'next_state',
                                                                'done'])
        self.seed = random.seed(seed)
        
    def add(self, state, action, reward, next_state, done):
        """
            Add an experience to the buffer
            
            Note: No need to remove the oldest experience from the queue as
            the deque class handles this automatically
        """
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)
        
    def sample(self):
        """
            Randomly sample a collection of experiences from the buffer
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([exp.state for exp in experiences if exp is not None])).float()
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences if exp is not None])).long()
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).float()
        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in experiences if exp is not None])).float()
        dones = torch.from_numpy(np.vstack([exp.done for exp in experiences if exp is not None]).astype(np.uint8)).float()
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)