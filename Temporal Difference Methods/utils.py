import os
import sys
from datetime import datetime

import gym
from gym import wrappers

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

class FeatureTransformer:
    def __init__(self, env):
        obs_examples = np.random.random((20000, 4))*2-1
        scaler = StandardScaler()
        scaler.fit(obs_examples)
        
        featurizer = FeatureUnion([('rbf1', RBFSampler(gamma=0.05, n_components=1000)),
                                   ('rbf2', RBFSampler(gamma=0.1, n_components=1000)),
                                   ('rbf3', RBFSampler(gamma=0.5, n_components=1000)),
                                   ('rbf4', RBFSampler(gamma=1.0, n_components=1000))
                                  ])
        feature_examples = featurizer.fit_transform(scaler.transform(obs_examples))
        
        self.dimensions = feature_examples.shape[1]
        
        self.scaler = scaler
        self.featurizer = featurizer
        
    def transform(self, obs):
        scaled = self.scaler.transform(obs)
        return self.featurizer.transform(scaled)
        
        
def plot_running_avg(rewards):
    N = len(rewards)
    running_avg = np.empty(N)
    
    for t in range(N):
        running_avg[t] = rewards[max(0, t-100):t+1].mean()
    
    plt.plot(running_avg)
    plt.title("Running Average Score")
    plt.show()