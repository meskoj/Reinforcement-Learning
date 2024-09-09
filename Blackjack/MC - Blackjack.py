#!/usr/bin/env python
#
# Author: Marco Meschini
# Created: 17-08-2024
#

"""
MC - Blackjack.py

This module implements the Blackjack game using the OpenAI Gym environment and the algorithm used is Monte Carlo.
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

class MonteCarloAgentEpsilonGreedy:
    def __init__(
            self,
            env,
            discount_factor,
            exploration_rate):

        self.env = env
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        # Create a dictionary to store the Q-values
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.Returns = defaultdict(lambda: np.zeros(env.action_space.n))
        self.N = defaultdict(lambda: np.zeros(env.action_space.n))

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

    def update_Q_values(self, episode):
        """
        Update Q-values based on the episode.

        Args:
            episode: List of (state, action, reward) tuples.
        """
        G = 0
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            self.Returns[state][action] += G
            self.N[state][action] += 1
            # Update rule for Q-values
            self.Q[state][action] = self.Returns[state][action] / \
                self.N[state][action]

if __name__ == "__main__":
    env = gym.make('Blackjack-v1', natural=False, sab=False)

    # Create an instance of the MonteCarloAgent class
    agent = MonteCarloAgentEpsilonGreedy(
        env, discount_factor=1, exploration_rate=0.1)

    num_episodes = 1000000

    for e in range(num_episodes):

        episode = []
        terminated = False
        truncated = False
        # Choose initial state randomly
        state, info = env.reset()
        
        # Render the environment for the last 10 episodes
        if e + 10 == num_episodes:
            env = gym.make('Blackjack-v1', natural=False,
                           sab=False, render_mode='human')
            state, info = env.reset()
            env.render()

        # Loop for each episode
        while (not (truncated or terminated)):
            
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state
        agent.update_Q_values(episode)

        if (e % 10000 == 0):
            print(f'Episode {e}/{num_episodes}')

    pl.plot_policy(agent.Q)

    env.close()
