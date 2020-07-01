# Navigation
The task is to train an agent using deep Q-Learning to collect yellow bananas (+1) and avoid blue bananas (-1) in a Unity environment. The state space contains 37 values and there are 4 action values (forward, left, right, backward). The task is considered solved when the agent obtains an average of at least 13 points over 100 consecutive episodes.

## Dependencies
The first cell of the notebook will download all of the requisite packages.

To install the Unity environment locally:
	1) Follow the instructions for a minimal install of OpenAI Gym at:
		https://github.com/openai/gym
	2) Download the environment from one of the links below. You need only select the environment that matches your operating system:
    	- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    	- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    	- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    	- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
	3) Save the Banana.app file to the same root directory as the notebook
    


## Training the agent
Training the agent simply requires executing all of the cells in the notebook sequentially. Any hyper parameters may be changed if so desired prior to running a cell.

At the end of training a .pth file containing the weights of the trained agent and a graph of the scores will be saved to the root directory.