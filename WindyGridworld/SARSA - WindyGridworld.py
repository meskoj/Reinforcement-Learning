#!/usr/bin/env python
#
# Author: Marco Meschini
# Created: 06-09-2024
#

"""
Module Name: SARSA - Windy Gridworld

This module implements the SARSA algorithm to solve the Windy Gridworld problem.
"""
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

DIM1 = 10
DIM2 = 6


class windy_gridworld():
    def __init__(self, dim1, dim2, boundary):
        """
        Initializes the Windy Gridworld environment.

        Args:
            dim1 (int): Dimension of the grid along the x-axis.
            dim2 (int): Dimension of the grid along the y-axis.
            boundary (bool): If True the agent always remains in the grid, if False the agent can go out of the grid.
        """
        "NOTE: If boundary is False, pay attention to really have a solution"
        self.dim1 = dim1
        self.dim2 = dim2
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.start = (0, 3)
        self.goal = (7, 3)
        self.states = self.populate_states()
        self.boundary = boundary

    def populate_states(self):
        """Populates the states dictonary with all possible states in the gridworld.
        This function iterates over the dimensions of the grid defined by DIM1 and DIM2
        and appends each grid position as a tuple (x, y) to the states.

        Returns:
            dict: Dictionary containing all possible states in the gridworld.
        """
        states = []
        for x in range(self.dim1):
            for y in range(self.dim2):
                states.append((x, y))
        return states

    def move(self, state, action):
        """Moves the agent from the current state to a new state.

        Args:
            state (tuple): Current position of the agent in the grid as (x, y).
            action (str): Current action to be taken by the agent.

        Returns:
            tuple: New state of the agent after taking the action.
            int: Reward for the action taken.
            bool: True if the agent has reached the goal, False otherwise.
        """
        # Wind is applied from bottom to top, so decrement y by wind[x]
        x, y = state
        if (self.boundary):
            if (action == "north"):
                next_state = (x, y-1 - self.wind[x])
            elif (action == "south"):
                next_state = (x, y+1 - self.wind[x])
            elif (action == "east"):
                next_state = (x+1, y - self.wind[x])
            elif (action == "west"):
                next_state = (x-1, y - self.wind[x])

            # If the agent is pushed out of the grid by the wind, it remains on the edge
            if (next_state[1] < 0):
                next_state = (next_state[0], 0)
                return next_state, -1, False
            # If the agent goes out of the grid, it remains in the same state
            elif (next_state[0] < 0 or next_state[0] > self.dim1-1 or next_state[1] > self.dim2-1):
                next_state = state
                return next_state, -1, False

            elif (next_state == self.goal):
                reward = 10
                return next_state, reward, True
            else:
                reward = -1
                return next_state, reward, False

        else:
            if (action == "north"):
                next_state = (x, y-1 - self.wind[x])
            elif (action == "south"):
                next_state = (x, y+1 - self.wind[x])
            elif (action == "east"):
                next_state = (x+1, y - self.wind[x])
            elif (action == "west"):
                next_state = (x-1, y - self.wind[x])

            # Out of grid
            if (next_state[0] < 0 or next_state[0] > self.dim1-1 or next_state[1] < 0 or next_state[1] > self.dim2-1):
                reward = -2
                # Pay attention, if the agent goes out of the grid, in the update function (not terminated) is 0,
                # so the Q-value of the next state is 0. (Is possible to return whatever state)
                return (0, 0), reward, True

            elif (next_state == self.goal):
                reward = 10
                return next_state, reward, True
            else:
                reward = -1
                return next_state, reward, False

    def print_grid(self, current_position):
        """
        Prints the grid with the agent's current position.

        Args:
            current_position (tuple): Current position of the agent in the grid as (x, y).
        """

        grid = np.zeros((self.dim2, self.dim1), dtype=str)

        grid[self.goal[1], self.goal[0]] = 'G'
        grid[self.start[1], self.start[0]] = 'S'
        grid[current_position[1], current_position[0]] = 'A'
        print("\n")
        print(grid)


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
        self.Q = {}

        for state in (env.states):
            self.Q[state] = {}
            for action in ["north", "south", "east", "west"]:
                self.Q[state][action] = 0

    def choose_action(self, state):
        """Chooses the next action to be taken by the agent.

        Args:
            state (tuple): Current position of the agent in the grid as (x, y).

        Returns:
            str: Action to be taken by the agent.
        """
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(["north", "south", "east", "west"])
        else:
            action = max(self.Q[state], key=self.Q[state].get)
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

    def print_best_policy(self, wg):
        """
        Prints the grid with the best policy found by the agent

        Args:
            wg: Environment object.
        """
        # Find the best action for each state
        best_actions = {}
        for state in self.Q:
            best_action = max(self.Q[state], key=self.Q[state].get) if max(
                self.Q[state].values()) != 0 else None
            best_actions[state] = best_action

        # Print the grid with directions
        grid = np.zeros((wg.dim2, wg.dim1), dtype=str)
        for state in best_actions:
            x, y = state
            if state == wg.goal:
                grid[y, x] = "G"
            elif state == wg.start:
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
                    grid[y, x] = "N"

        print(grid)


def main():
    wg = windy_gridworld(DIM1, DIM2, boundary=True)
    agent = SARSA_Agent(wg)

    num_episodes = 10000
    n_steps = [0] * num_episodes  # Store the number of steps for each episode
    for episode in range(num_episodes):
        if (episode % 1000 == 0):
            print(f'Episode {episode}/{num_episodes}')
        terminated = False
        state = wg.start
        steps = 0

        while not terminated:
            action = agent.choose_action(state)
            next_state, reward, terminated = wg.move(state, action)
            steps += 1
            next_action = agent.choose_action(next_state)
            agent.update_Q(state, action, reward, next_state,
                           next_action, terminated)
            state = next_state
            # Print the last episode
            if episode == num_episodes-1:
                wg.print_grid(state)
        n_steps[episode] = steps

    # Print information about the performance of the agent and the best policy
    print("\n")
    agent.print_best_policy(wg)
    mean_steps = mean(n_steps[-100:])
    print(mean_steps)

    plt.figure(figsize=(10, 5))
    plt.plot(range(100, num_episodes),
             n_steps[100:], label='Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Number of Steps')
    plt.title('Number of Steps per Episode in SARSA Windy Gridworld')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
