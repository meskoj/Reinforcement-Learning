#!/usr/bin/env python
#
# Author: Marco Meschini
# Created: 09-09-2024
#

"""
Module Name: CliffWalking.py

This module implements SARSA algorithm to solve the Cliff Walking problem.
"""

import gymnasium as gym
import numpy as np
from collections import defaultdict
import plots as pl


class SARSA_Agent():
    def __init__(self, env, gamma=1, epsilon=0.1, alpha=0.5):
        """
        Initializes the SARSA agent.

        Args:
            env: Environment object. 
            gamma (float, optional): Discount factor. Defaults to 1.
            epsilon (float, optional): Exploration rate. Defaults to 0.1.
            alpha (float, optional): Learning rate. Defaults to 0.5.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.env = env
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def get_action(self, state):
        """
        (Epsilon-Greedy policy)
        Get the action with the highest Q-value for a given observation with probability 1-epsilon.
        Otherwise, choose a random action.  

        Args:
            state: The observation for which the action is to be determined.

        Returns:
            action: The action with the highest Q-value with probability 1-epsilon, otherwise a random action. 
        """

        if np.random.rand() < self.epsilon:
            # Choose a random action
            action = self.env.action_space.sample()
        else:
            # Choose the action with the highest Q-value
            action = int(np.argmax(self.Q[state]))

        return action

    def update_Q(self, state, action, reward, next_state, next_action, terminated):
        """Updates the Q-value for the state-action pair.

        Args:
            state (tuple): Current position of the agent in the grid as (x, y).
            action (str): Action taken by the agent.
            reward (int): Reward for the action taken.
            next_state (tuple): New position of the agent in the grid as (x, y).
            next_action (str): Next action to be taken by the agent.
        """
        self.Q[state][action] = self.Q[state][action] + self.alpha * \
            (reward + self.gamma * (not terminated) *
             self.Q[next_state][next_action] - self.Q[state][action])


def main():
    env = gym.make('CliffWalking-v0')
    agent = SARSA_Agent(env)

    num_episodes = 20000
    n_steps = [0] * num_episodes  # Store the number of steps for each episode
    rewards = [0] * num_episodes  # Store the rewards for each episode

    for episode in range(num_episodes):
        
        if episode % 1000 == 0:
            print(f"Episode {episode}/{num_episodes}")

        terminated = False
        truncated = False
        # Choose initial state randomly
        state, info = env.reset()

        # Render the environment for the last episode
        if episode + 1 == num_episodes:
            env = gym.make('CliffWalking-v0', render_mode='human')
            state, info = env.reset()
            env.render()
        
        steps = 0
        # Loop for each episode
        while (not (truncated or terminated)):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_action = agent.get_action(next_state)
            agent.update_Q(state, action, reward, next_state,
                           next_action, terminated)
            state = next_state
            steps += 1
            rewards[episode] += reward
        n_steps[episode] = steps
    
    # Plot the rewards obtained by the agent at each episode    
    pl.plot_rewards_overtime(rewards, range=(0, num_episodes))
    # Plot the number of steps taken by the agent at each episode for the last 100 episodes
    pl.plot_steps_overtime(n_steps, range=(num_episodes-100, num_episodes))
    
    env.close()

if __name__ == "__main__":
    main()
