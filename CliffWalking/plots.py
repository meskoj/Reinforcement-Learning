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
    