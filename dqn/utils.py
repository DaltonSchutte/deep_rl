import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
        
        
def plot_running_avg(rewards, window=100):
    """
        Plots the running average of desired window size
        ARGS:
        rewards(list):= list of episodic rewards
        window(int):= size of the window the average will be computed over
    """
    N = len(rewards)
    running_avg = np.empty(N)
    
    for t in range(N):
        running_avg[t] = rewards[max(0, t-window):t+1].mean()
    
    plt.plot(running_avg)
    plt.title("Running Average ({}) Score".format(window))
    plt.show()
    

def show_off(agent, path, env, episodes):
    """
        Renders an episode of the trained agent playing in the environment
        ARGS:
        agent(Agent):= agent object that will interact with the environment
        path(str):= location of the saved state_dict
        episodes(int):= number of episodes for the agent to play
    """
    #Load state_dict
    agent.online_net.load_state_dict(torch.load(path))
    
    #Loop to play n episodes
    for i in range(episodes):
        state = env.reset()
        for n in range(200):
            action = agent.act(state)
            env.render()
            state, reward, done, _ = env.step(action)
            if done:
                break
              
    env.close()