#!/usr/bin/env python
#
# Author: Marco Meschini
# Created: 2024-09-09
#

"""
Module Name: plots.py

This module implements functions to plot the results of the various algorithms.
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def plot_policy(Q):
    # Initialize Q-tables for usable and non-usable ace
    q_table_usable_ace = np.full((18, 11), -np.inf)
    q_table_non_usable_ace = np.full((18, 11), -np.inf)
    action_table_usable_ace = np.full(
          (18, 11), 'stick')  # Default action is 'stick'
    action_table_non_usable_ace = np.full((18, 11), 'stick')

    # Fill the Q-tables with the maximum Q-value for each state
    for state, actions in Q.items():
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
