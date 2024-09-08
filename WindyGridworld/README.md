This repository contains the windy gridworld example from Sutton and Barto book.
The text of the exercise can be found in the figure below:

![alt text](https://github.com/meskoj/Reinforcement-Learning/blob/main/WindyGridworld/WindyGridworld.PNG)

# Code observation
The solution presented assumes a simple scenario where the gridworld has boundaries, preventing the agent from moving beyond the edges. While this setup is straightforward, I propose an alternative environment without
such boundaries. In this boundaryless gridworld, the agent struggles more in discovering the optimal policy.

If you try this environment, please pay attention to put the agent in the possibility to find a solution. In many configurations, the absence of boundaries can result in the agent struggling to converge,
especially if no clear path to the goal is present.
