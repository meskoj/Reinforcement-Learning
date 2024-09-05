#!/usr/bin/env python
#
# Author: Marco Meschini
# Created: 2024-08-17
#

"""
Blackjack.py

This module implements the Blackjack game using the OpenAI Gym environment.
"""

"""NOTE: The game of Blackjack starts with the player having 2 cards and the dealer with two cards with one faced down, while other faced up. Now the player can have the sum of those cards from 2 to 22. Minimum in the case where they receive two aces and keep there values as 1 each or maximum when the keep there values to be 11 each.
Note: aces can have a value of either 1 or 11.

Now when the sum is 22, and the player chooses to hit he may get a card with value 10, resulting in a sum of 32, and thus loosing the game.
Also the dealer can have only cards with value 1-10 (where 1 is ace), which accounts to 10 possible values.

But it seems the programmers also kept the not possible states of 0 sum or 0 value card with dealer.
Thus the observation space being a tuple (Discrete(32), Discrete(11), Discrete(2)).."""
"""0: Stick

1: Hit"""



import gymnasium as gym
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

class MonteCarloAgentEpsilonGreedy:
    def __init__(
            self,
            env,
            discount_factor,
            epsilon):

        self.env = env
        self.gamma = discount_factor
        self.epsilon = epsilon  # Exploration rate
        # Create a dictionary to store the Q-values
        self.Q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.Returns = defaultdict(lambda: np.zeros(env.action_space.n))
        self.N = defaultdict(lambda: np.zeros(env.action_space.n))
        """# Find the dimensions of the observation space and action space
        obs_space_ranges = [range(space.n) for space in env.observation_space]
        action_space_range = range(env.action_space.n)

        # Create a dictionary with (state, action) = 0
        # NOTE: This could be done more efficiently by using a defaultdict, but this is done in this way for clarity 
        for obs1 in obs_space_ranges[0]:
            for obs2 in obs_space_ranges[1]:
                for obs3 in obs_space_ranges[2]:
                    for action in action_space_range:
                        self.Q_values[((obs1, obs2, obs3), action)] = 0"""

    def get_action(self, obs):
        """
        Get the action with the highest Q-value for a given observation. (Epsilon-Greedy policy)

        Args:
            obs: The observation for which the action is to be determined.

        Returns:
            action: The action with the highest Q-value with probability 1-epsilon, otherwise a random action. 
        """

        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample() # Choose a random action
        else:
            action = int(np.argmax(self.Q_values[obs])) # Choose the action with the highest Q-value

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
            self.Q_values[state][action] = self.Returns[state][action] / self.N[state][action]

    def plot_Q_values(self):
        # Initialize Q-tables for usable and non-usable ace
        q_table_usable_ace = np.full((18, 11), -np.inf)
        q_table_non_usable_ace = np.full((18, 11), -np.inf)
        action_table_usable_ace = np.full((18, 11), 'stick')  # Default action is 'stick'
        action_table_non_usable_ace = np.full((18, 11), 'stick')

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
                            action_table_usable_ace[row, col] = 'stick' if best_action == 0 else 'hit'
                else:
                    if q_value > q_table_non_usable_ace[row, col]:
                        q_table_non_usable_ace[row, col] = round(q_value, 2)
                        action_table_non_usable_ace[row, col] = 'stick' if best_action == 0 else 'hit'
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
                color = 'lightcoral' if action_table_usable_ace[i, j] == 'hit' else 'lightblue'
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
        ax[0].invert_yaxis()  # Reverse the y-axis
        ax[0].set_title('Q-values with Usable Ace')

        # Table for non-usable ace
        cell_text_non_usable_ace = []
        cell_colors_non_usable_ace = []
        for i in range(18):
            row_text = []
            row_colors = []
            for j in range(10):
                row_text.append(f'{q_table_non_usable_ace[i, j]:.2f}')
                color = 'lightcoral' if action_table_non_usable_ace[i, j] == 'hit' else 'lightblue'
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
        ax[1].invert_yaxis()  # Reverse the y-axis
        ax[1].set_title('Q-values without Usable Ace')

        plt.show()
    def plot_Q_values2(self):
        def create_grids(usable_ace=False):

            # convert our state-action values to state values
            # and build a policy dictionary that maps observations to actions
            state_value = defaultdict(float)
            policy = defaultdict(int)
            for obs, action_values in self.Q_values.items():
                state_value[obs] = float(np.max(action_values))
                policy[obs] = int(np.argmax(action_values))

            player_count, dealer_count = np.meshgrid(
                # players count, dealers face-up card
                np.arange(12, 22),
                np.arange(1, 11),
            )

            # create the value grid for plotting
            value = np.apply_along_axis(
                lambda obs: state_value[(obs[0], obs[1], usable_ace)],
                axis=2,
                arr=np.dstack([player_count, dealer_count]),
            )
            value_grid = player_count, dealer_count, value

            # create the policy grid for plotting
            policy_grid = np.apply_along_axis(
                lambda obs: policy[(obs[0], obs[1], usable_ace)],
                axis=2,
                arr=np.dstack([player_count, dealer_count]),
            )
            return value_grid, policy_grid


        def create_plots(value_grid, policy_grid, title: str):
            """Creates a plot using a value and policy grid."""
            # create a new figure with 2 subplots (left: state values, right: policy)
            player_count, dealer_count, value = value_grid
            fig = plt.figure(figsize=plt.figaspect(0.4))
            fig.suptitle(title, fontsize=16)

            # plot the state values
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            ax1.plot_surface(
                player_count,
                dealer_count,
                value,
                rstride=1,
                cstride=1,
                cmap="viridis",
                edgecolor="none",
            )
            plt.xticks(range(12, 22), range(12, 22))
            plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
            ax1.set_title(f"State values: {title}")
            ax1.set_xlabel("Player sum")
            ax1.set_ylabel("Dealer showing")
            ax1.zaxis.set_rotate_label(False)
            ax1.set_zlabel("Value", fontsize=14, rotation=90)
            ax1.view_init(20, 220)

            # plot the policy
            fig.add_subplot(1, 2, 2)
            ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
            ax2.set_title(f"Policy: {title}")
            ax2.set_xlabel("Player sum")
            ax2.set_ylabel("Dealer showing")
            ax2.set_xticklabels(range(12, 22))
            ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

            # add a legend
            legend_elements = [
                Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
                Patch(facecolor="grey", edgecolor="black", label="Stick"),
            ]
            ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
            return fig


        # state values & policy with usable ace (ace counts as 11)
        value_grid, policy_grid = create_grids()
        fig1 = create_plots(value_grid, policy_grid, title="Without usable ace")
        plt.show()


if __name__ == "__main__":
    env = gym.make('Blackjack-v1', natural=False, sab=False)

    # Create an instance of the MonteCarloAgent class
    agent = MonteCarloAgentEpsilonGreedy(
        env, discount_factor=1, epsilon=0.1)

    num_episodes = 1000000

    for e in range(num_episodes):

        episode = []
        terminated = False
        truncated = False
        # Choose initial state randomly
        observation, info = env.reset()

        while (not terminated and not truncated):  # Loop for each episode

            action = agent.get_action(observation)

            next_obs, reward, terminated, truncated, info = env.step(action)
            episode.append((observation, action, reward))
        agent.update_Q_values(episode)
    
   
    agent.plot_Q_values2()

    env.close()
