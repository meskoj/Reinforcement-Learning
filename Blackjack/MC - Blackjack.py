#!/usr/bin/env python
#
# Author: Marco Meschini
# Created: 17-08-2024
#

"""
Blackjack.py

This module implements the Blackjack game using the OpenAI Gym environment.
"""

"""NOTE: The game of Blackjack starts with the player having 2 cards and the dealer with two cards with one faced down,
while other faced up. Now the player can have the sum of those cards from 2 to 22. Minimum in the case where they receive
two aces and keep there values as 1 each or maximum when the keep there values to be 11 each.
Note: aces can have a value of either 1 or 11.

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
import matplotlib.pyplot as plt
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
        self.Q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.Returns = defaultdict(lambda: np.zeros(env.action_space.n))
        self.N = defaultdict(lambda: np.zeros(env.action_space.n))

    def get_action(self, obs):
        """
        Get the action with the highest Q-value for a given observation. (Epsilon-Greedy policy)

        Args:
            obs: The observation for which the action is to be determined.

        Returns:
            action: The action with the highest Q-value with probability 1-epsilon, otherwise a random action. 
        """

        if np.random.rand() < self.epsilon:
            # Choose a random action
            action = self.env.action_space.sample()
        else:
            # Choose the action with the highest Q-value
            action = int(np.argmax(self.Q_values[obs]))

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
            self.Q_values[state][action] = self.Returns[state][action] / \
                self.N[state][action]

    def plot_Q_values(self):
        # Initialize Q-tables for usable and non-usable ace
        q_table_usable_ace = np.full((18, 11), -np.inf)
        q_table_non_usable_ace = np.full((18, 11), -np.inf)
        action_table_usable_ace = np.full(
            (18, 11), 'hit')  # Default action is 'stick'
        action_table_non_usable_ace = np.full((18, 11), 'hit')

        # Fill the Q-tables with the maximum Q-value for each state
        for state, actions in self.Q_values.items():
            player_sum, dealer_card, usable_ace = state
            if 4 <= player_sum <= 21 and 1 <= dealer_card <= 10:
                row = player_sum - 4
                col = dealer_card - 1
                best_action = np.argmax(actions)
                q_value = np.max(actions)
                if usable_ace:
                    if q_value > q_table_usable_ace[row, col]:
                        q_table_usable_ace[row, col] = round(q_value, 2)
                        action_table_usable_ace[row,
                                                col] = 'stick' if best_action == 0 else 'hit'
                else:
                    if q_value > q_table_non_usable_ace[row, col]:
                        q_table_non_usable_ace[row, col] = round(q_value, 2)
                        action_table_non_usable_ace[row,
                                                    col] = 'stick' if best_action == 0 else 'hit'
            # Create a table for usable ace
        fig, ax = plt.subplots(2, 1, figsize=(30, 10))

        # Table for usable ace
        cell_text_usable_ace = []
        cell_colors_usable_ace = []
        for i in range(18):
            row_text = []
            row_colors = []
            for j in range(10):
                row_text.append(f'{q_table_usable_ace[i, j]:.2f}')
                color = 'lightcoral' if action_table_usable_ace[i,
                                                                j] == 'hit' else 'lightblue'
                row_colors.append(color)
            cell_text_usable_ace.append(row_text)
            cell_colors_usable_ace.append(row_colors)

        ax[0].axis('tight')
        ax[0].axis('off')
        table_usable_ace = ax[0].table(cellText=cell_text_usable_ace,
                                       cellColours=cell_colors_usable_ace,
                                       rowLabels=np.arange(4, 22),
                                       colLabels=np.arange(1, 11),
                                       cellLoc='center',
                                       loc='center')
        ax[0].set_title('Q-values with Usable Ace')

        # Table for non-usable ace
        cell_text_non_usable_ace = []
        cell_colors_non_usable_ace = []
        for i in range(18):
            row_text = []
            row_colors = []
            for j in range(10):
                row_text.append(f'{q_table_non_usable_ace[i, j]:.2f}')
                color = 'lightcoral' if action_table_non_usable_ace[i,
                                                                    j] == 'hit' else 'lightblue'
                row_colors.append(color)
            cell_text_non_usable_ace.append(row_text)
            cell_colors_non_usable_ace.append(row_colors)

        ax[1].axis('tight')
        ax[1].axis('off')
        table_non_usable_ace = ax[1].table(cellText=cell_text_non_usable_ace,
                                           cellColours=cell_colors_non_usable_ace,
                                           rowLabels=np.arange(4, 22),
                                           colLabels=np.arange(1, 11),
                                           cellLoc='center',
                                           loc='center')
        ax[1].set_title('Q-values without Usable Ace')

        plt.show()


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
        obs, info = env.reset()

        # Loop for each episode
        while (not (truncated or terminated)):
            action = agent.get_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            episode.append((obs, action, reward))
            obs = next_obs
        agent.update_Q_values(episode)

        if (e % 10000 == 0):
            print(f'Episode {e}/{num_episodes}')

    agent.plot_Q_values()

    env.close()
