#!/usr/bin/env python
#
# Author: Marco Meschini
# Created: 08-09-2024
#

"""
Q-Learning - Blackjack.py

This module implements the Blackjack game using the OpenAI Gym environment and the algorithm used is Q-learning.
"""

"""NOTE: The game of Blackjack starts with the player having 2 cards and the dealer with two cards with one faced down, while other faced up.
Now the player can have the sum of those cards from 2 to 22. Minimum in the case where they receive two aces and keep there values as 1 each 
or maximum when the keep there values to be 11 each.

Now when the sum is 22, and the player chooses to hit he may get a card with value 10, resulting in a sum of 32, and thus loosing the game.
Also the dealer can have only cards with value 1-10 (where 1 is ace), which accounts to 10 possible values.

But it seems the programmers also kept the not possible states of 0 sum or 0 value card with dealer.
Thus the observation space being a tuple (Discrete(32), Discrete(11), Discrete(2)).."""
"""Possible actions:
    0: Stick
    1: Hit
"""




import gymnasium as gym
from collections import defaultdict
import numpy as np
import plots as pl

class QlearningAgent:
    def __init__(
            self,
            env,
            discount_factor,
            exploration_rate,
            learning_rate):

        self.env = env
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.alpha = learning_rate
        # Create a dictionary to store the Q-values
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

    def update_Q(self, state, action, reward, next_state, done):
        """
        Update Q-values based on the episode.

        Args:
            episode: List of (state, action, reward) tuples.
        """
        self.Q[state][action] = self.Q[state][action] + self.alpha * \
            (reward + self.gamma *
             (not done) * np.max(self.Q[next_state]) - self.Q[state][action])


if __name__ == "__main__":
    env = gym.make('Blackjack-v1', natural=False, sab=False)

    # Create an instance of the Q-learning agent class
    agent = QlearningAgent(
        env, discount_factor=1, exploration_rate=0.95, learning_rate=0.01)

    num_episodes = 500000

    for episode in range(num_episodes):

        terminated = False
        truncated = False
        # Choose initial state randomly
        state, info = env.reset()

        # Render the environment for the last 10 episodes
        if episode + 10 == num_episodes:
            env = gym.make('Blackjack-v1', natural=False,
                           sab=False, render_mode='human')
            state, info = env.reset()
            env.render()

        # Loop for each episode
        while (not (truncated or terminated)):

            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.update_Q(state, action, reward, next_state, done)
            state = next_state

        if (episode % 10000 == 0):
            print(f'Episode {episode}/{num_episodes}')

    pl.plot_policy(agent.Q)

    env.close()
