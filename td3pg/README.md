# Continuous Control
The task is to train an agent using deep RL for continuous control to move robotic arms so the end is within a moving target region. The state space contains 37 values and there are 4 action values (forward, left, right, backward). The task is considered solved when the agent obtains an average of at least 13 points over 100 consecutive episodes.

## Dependencies
The first cell of the notebook will download all of the requisite packages.

To install the Unity environment locally:
	1) Follow the instructions for a minimal install of OpenAI Gym at:
		https://github.com/openai/gym
	2) Download the environment from one of the links below. You need only select the environment that matches your operating system:
    Version 1: One Agent
    -Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    -Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    -Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
    Version 2: Twenty Agents
    -Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    -Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    -Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
	3) Unzip the file in the root directory of the repository.

## Training the Agent
Training the agent simply requires executing all of the cells in the notebook sequentially. Any hyperparameters may be changed if so desired prior to running a cell. Hyperparameters for the TD3PG agent must be changed in the agent.py file.

At the end of training a .pth file containing the weights for the agent models and a graph of the scores will be saved to the root directory.

## Viewing a Trained Agent
An instance of the Agent class can be instantiated and the state dict for each model attribute loaded from the corresponding pth weight file in the weights directory. The Agent can then be used to interact with the Unity Reacher environment to control 20 arms simultaneously.
