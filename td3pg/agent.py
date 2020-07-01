import copy
import random
from collections import deque, namedtuple

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from models import Actor, Critic

DEFAULTS = False 

if DEFAULTS: #Hyperparameters used in paper
    CAPACITY = 1000000
    ACT_LR = 0.0001
    CRT_LR = 0.001
    GAMMA = 0.99
    TAU = 0.001
    BATCH_SIZE = 64
    MU = 0.0
    THETA = 0.15
    SIGMA = 0.2
    FC1 = 400
    FC2 = 300
else: #Custom hyperparameters
    CAPACITY = int(2e5) #Best 1e5 #20 arms for 1000 steps -> 2e4 experiences / episode, will use min of 2e4
    ACT_LR = 0.0003  
    CRT_LR = 0.003 
    WEIGHT_DECAY = 0.00
    GAMMA = 0.99
    TAU = 0.005
    BATCH_SIZE = 512 #Best 512
    MU = 0.0
    THETA = 0.15
    SIGMA = 0.2
    FC1 = 256
    FC2 = 128

UPDATE_EVERY = 10
UPDATE_TIMES = 15
CLIP = 1.0 #Best 1.0
    
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    #####################
    ## Agent for TD3PG ##
    #####################
    def __init__(self, state_size, action_dim, seed):
        self.state_size = state_size
        self.action_dim = action_dim
        self.t = 0

        #Policy networks
        self.act_local = Actor(state_size, action_dim, FC1, FC2, seed).to(DEVICE)
        self.act_target = Actor(state_size, action_dim, FC1, FC2, seed).to(DEVICE)
        self.act_target.load_state_dict(self.act_local.state_dict()) #Sets the weights per the DDPG paper
        self.act_optim = optim.Adam(self.act_local.parameters(), lr=ACT_LR, amsgrad=True)

        #Value networks
        self.crt_local_1 = Critic(state_size, action_dim, FC1, FC2, seed).to(DEVICE)
        self.crt_local_2 = Critic(state_size, action_dim, FC1, FC2, seed).to(DEVICE)
        self.crt_target_1 = Critic(state_size, action_dim, FC1, FC2, seed).to(DEVICE)
        self.crt_target_2 = Critic(state_size, action_dim, FC1, FC2, seed).to(DEVICE)
        self.crt_target_1.load_state_dict(self.crt_local_1.state_dict()) #Sets the weights per the TD3PG Paper
        self.crt_target_2.load_state_dict(self.crt_local_2.state_dict()) #Sets the weights per the TD3PG Paper
        self.crt_optim_1 = optim.Adam(self.crt_local_1.parameters(), lr=CRT_LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
        self.crt_optim_2 = optim.Adam(self.crt_local_2.parameters(), lr=CRT_LR, weight_decay=WEIGHT_DECAY, amsgrad=True)

        #Create noise
        self.noise = OrnsteinUhlenbeck(action_dim, MU, THETA, SIGMA, seed)

        self.memory = ReplayMemory(CAPACITY, BATCH_SIZE, seed)

        self.seed = random.seed(seed)
    
    def act(self, state, epsilon):
        state = torch.from_numpy(state).float().to(DEVICE)
        self.act_local.eval()
        with torch.no_grad():
            action = self.act_local(state).cpu().data.numpy()
        self.act_local.train()
        noise = epsilon * self.noise.sample()
        action += noise
        return np.clip(action, -1.0, 1.0)
    
    def step(self, states, actions, rewards, next_states, dones):
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            self.memory.add_exp(s, a, r, ns, d)
        
        self.t += 1
        
        if (self.t % UPDATE_EVERY == 0) and (len(self.memory) >= BATCH_SIZE):
            for _ in range(UPDATE_TIMES):
                exps = self.memory.sample()
                self.learn(exps)
            self.t = 0
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        
        #### Local Critic Updates ####
        next_actions = self.act_target(next_states) + torch.empty(self.action_dim).normal_(0, 0.2).to(DEVICE)
        next_actions = torch.clamp(next_actions, -1, 1)
        Q_targets_next = torch.min(self.crt_target_1(next_states, next_actions),
                                   self.crt_target_2(next_states, next_actions)
                                  )
        Q_targets = rewards + (GAMMA * (1 - dones) * Q_targets_next)
        Q_expected_1 = self.crt_local_1(states, actions)
        Q_expected_2 = self.crt_local_2(states, actions)
        
        crt_loss = F.mse_loss(Q_targets, Q_expected_1) + F.mse_loss(Q_targets, Q_expected_2)
        
        #Critic Updates
        self.crt_optim_1.zero_grad()
        self.crt_optim_2.zero_grad()
        crt_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.crt_local_1.parameters(), CLIP)
        torch.nn.utils.clip_grad_norm_(self.crt_local_2.parameters(), CLIP)
        self.crt_optim_1.step()
        self.crt_optim_2.step()
        
        #### Local Actor Updates ####
        if self.t % 2 == 0:
            pred_actions = self.act_local(states)
            act_loss = -self.crt_local_1(states, pred_actions).mean()
        
            self.act_optim.zero_grad()
            act_loss.backward()
            self.act_optim.step()
        
        #Target Network Updates
            self.soft_update(self.crt_local_1, self.crt_target_1, TAU)
            self.soft_update(self.crt_local_2, self.crt_target_2, TAU)
            self.soft_update(self.act_local, self.act_target, TAU)
    
    def soft_update(self, local_net, target_net, tau):
        for l_param, t_param in zip(local_net.parameters(), target_net.parameters()):
            t_param.data.copy_((tau * l_param.data) + ((1 - tau) * t_param.data))
    
    def reset(self):
        self.t = 0
        self.noise.reset()
        
        
class ReplayMemory():
    def __init__(self, capacity, batch_size, seed):
        self.n = capacity
        self.bs = batch_size
        self.exp = namedtuple("Experience", field_names=['state','action','reward','next_state','done'])
        
        self.memory = deque(maxlen=capacity)
        
        self.seed = random.seed(seed)
        
    def sample(self):
        exps = random.sample(self.memory, k=self.bs)
        
        states = torch.from_numpy(np.vstack([e.state for e in exps if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in exps if e is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in exps if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in exps if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in exps if e is not None]).astype(np.uint8)).float().to(DEVICE)
        
        return (states, actions, rewards, next_states, dones)
    
    def add_exp(self, state, action, reward, next_state, done):
        exp = self.exp(state, action, reward, next_state, done)
        self.memory.append(exp)
        
    def __len__(self):
        return len(self.memory)
    
    
class OrnsteinUhlenbeck():
    def __init__(self, size, mu, sigma, theta, seed):
        self.size = size
        self.mu = mu * np.ones(size)
        self.sigma = sigma
        self.theta = theta
        self.seed = random.seed(seed)
        
        self.reset()
    
    def reset(self):
        self.state = copy.copy(self.mu)
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(0, 0.1, size=self.size)
        self.state = x + dx
        return self.state
    