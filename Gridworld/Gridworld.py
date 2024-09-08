#!/usr/bin/env python
#
# Author: Marco Meschini
# Created: 02-08-2024
#

"""
Module Name: GridWorld.py

This module implements the Dynamic Programming algorithm to solve the Gridworld problem.
"""
import pandas as pd  # Used only to format the final result

DIM1 = 5
DIM2 = 5
GAMMA = 0.9


def populate_states():
    """Populates the states dictonary with all possible states in the gridworld.
    This function iterates over the dimensions of the grid defined by DIM1 and DIM2
    and appends each grid position as a tuple (x, y) to the states.

    Returns:
        list: Dictionary containing all possible states in the gridworld.
    """
    global DIM1, DIM2
    states = []
    for x in range(0, DIM1):
        for y in range(0, DIM2):
            states.append((x, y))
    return states


def move(state, action):
    """Moves the agent from the current state to a new state.

    Args:
        state (tuple): Current position of the agent in the grid as (x, y).
        action (list): Current action to be taken by the agent.

    Returns:
        int: Reward for the action taken.
        list: New state of the agent after taking the action.
    """
    x, y = state
    if state == (1, 0):  # If state is (1,0), move to (1,4) and add 10 to the reward
        reward = 10
        return reward, (1, 4)
    if state == (3, 0):  # If state is (3,0), move to (3,2) and add 5 to the reward
        reward = 5
        return reward, (3, 2)

    # If the action allows to remain inside the gridworld, then reward is 0.
    # Otherwise, reward is -1
    if (action == "north" and y > 0):
        reward = 0
        return reward, (x, y-1)
    elif (action == "south" and y < DIM2-1):
        reward = 0
        return reward, (x, y+1)
    elif (action == "east" and x < DIM1-1):
        reward = 0
        return reward, (x+1, y)
    elif (action == "west" and x > 0):
        reward = 0
        return reward, (x-1, y)
    else:
        reward = -1
        return reward, state


def policy_evaluation(actions, states, value_function):
    """
    Implement the policy evaluation algorithm to calculate the value function for the given policy.

    Args:
        actions (list): List of possible actions that the agent can take.
        states (list): List of all possible states in the gridworld.
        value_function (list): List of values for each state in the gridworld.

    Returns:
        list: Value function for the given policy.
    """
    THRESHOLD = 0.01
    global GAMMA
    first_iteration = True
    while True:
        delta = 0
        old_value_function = value_function.copy()
        for state in states:
            value_function[state] = 0
            for action in actions:
                reward, new_state = move(state, action)
                value_function[state] += 0.25 * \
                    (reward + GAMMA * old_value_function[new_state])
            delta = max(delta, abs(
                old_value_function[state] - value_function[state]))

        if delta < THRESHOLD and first_iteration == False:
            break
        first_iteration = False
    return value_function


def policy_improvement(actions, states, value_function, policy):
    """
    Implement the policy improvement algorithm to update the policy.

    Args:
        actions (list): List of possible actions that the agent can take.
        states (list): List of all possible states in the gridworld.
        value_function (list): List of values for each state in the gridworld.
        policy (list): List of actions for each state, based on the current policy.

    Returns:
        list: List of actions for each state, based on the updated policy.
        bool: True if the policy is stable, False otherwise.
    """
    global GAMMA
    policy_stable = True

    for state in states:
        old_action = policy[state]
        action_values = {}
        for action in actions:
            reward, new_state = move(state, action)
            action_values[action] = reward + GAMMA * value_function[new_state]
        best_action = max(action_values, key=action_values.get)
        policy[state] = best_action
        if old_action != best_action:
            policy_stable = False
    return policy, policy_stable


def main():
    global DIM1, DIM2, GAMMA
    actions = ["north", "south", "east", "west"]
    states = populate_states()
    value_function = {state: 0 for state in states}
    # Initialize with the first action
    policy = {state: actions[0] for state in states}
    policy_stable = False

    while not policy_stable:
        value_function = policy_evaluation(actions, states, value_function)
        policy, policy_stable = policy_improvement(
            actions, states, value_function, policy)

##### PLOTTING RESULT #####
    value_data = []
    for y in range(DIM2):
        row = []
        for x in range(DIM1):
            state = (x, y)
            row.append(round(value_function[state], 1))
        value_data.append(row)

    policy_data = []
    for y in range(DIM2):
        row = []
        for x in range(DIM1):
            state = (x, y)
            row.append(policy[state])
        policy_data.append(row)

    value_df = pd.DataFrame(value_data)
    policy_df = pd.DataFrame(policy_data)

    print("Value Function:")
    print(value_df.to_string(index=False, header=False))

    print("\nPolicy:")
    print(policy_df.to_string(index=False, header=False))


if __name__ == "__main__":
    main()
