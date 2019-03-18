import numpy as np
import matplotlib
import matplotlib.pyplot as plt
        
        
def plot_running_avg(rewards):
    N = len(rewards)
    running_avg = np.empty(N)
    
    for t in range(N):
        running_avg[t] = rewards[max(0, t-100):t+1].mean()
    
    plt.plot(running_avg)
    plt.title("Running Average Score")
    plt.show()