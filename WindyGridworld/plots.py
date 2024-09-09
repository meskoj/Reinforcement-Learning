#!/usr/bin/env python
#
# Author: Marco Meschini
# Created: 2024-09-09
#

"""
Module Name: plots.py

This module implements functions to plot the results of the various algorithms.
"""
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean

def plot_rewards_overtime(rewards, range=None):
    """
    Plots the rewards obtained by the agent over time.

    Args:
        rewards (list): List containing the sum of rewards obtained by the agent at each episode.
        range (tuple, optional): Range of episodes to plot. If None, the entire list is plotted. Defaults to None.
    """
    if range is None:
        range = (0, len(rewards))
    plt.figure(figsize=(10, 5))
    plt.plot(rewards[range[0]:range[1]], label="Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards over time")
    plt.legend()
    plt.show()

def plot_steps_overtime(steps, range=None):
    """
    Plots the number of steps taken by the agent to reach the goal at each episode.

    Args:
        steps (list): List containing the number of steps taken by the agent at each episode.
        range (tuple, optional): Range of episodes to plot. If None, the entire list is plotted. Defaults to None.
    """
    if range is None:
        range = (0, len(steps))
    plt.figure(figsize=(10, 5))
    plt.plot(steps[range[0]:range[1]], label="Steps")
    # Calculate and plot the mean of the steps
    mean_steps = mean(steps)
    plt.axhline(y=mean_steps, color='r', linestyle='--',
                label=f'Mean Steps: {mean_steps:.2f}')
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Steps over time")
    plt.legend()
    plt.show()
    
def print_best_policy(gw, Q):
    """
    Prints the grid with the best policy found by the agent. The result is a grid containing arrows indicating the best policy for each state.
    If the agent has not learned a policy for a state, the state is marked with a "#".
    Grid legend:
        - "↑": Move north.
        - "↓": Move south.
        - "→": Move east.
        - "←": Move west.
        - "S": Start position.
        - "G": Goal position.
        - "#": No policy learned for this state.

    Args:
        gw: Environment object.
        Q: Dictionary containing the Q-values for each state-action pair.
    """
    # Find the best action for each state
    best_actions = {}
    for state in Q:
        best_action = max(Q[state], key=Q[state].get) if max(
            Q[state].values()) != 0 else None
        best_actions[state] = best_action

    # Print the grid with directions
    grid = np.zeros((gw.dim2, gw.dim1), dtype=str)
    for state in best_actions:
        x, y = state
        if state == gw.goal:
            grid[y, x] = "G"
        elif state == gw.start:
            grid[y, x] = "S"
        else:
            action = best_actions[state]
            if action == "north":
                grid[y, x] = "↑"
            elif action == "south":
                grid[y, x] = "↓"
            elif action == "east":
                grid[y, x] = "→"
            elif action == "west":
                grid[y, x] = "←"
            else:
                grid[y, x] = "#"

    print(grid)

