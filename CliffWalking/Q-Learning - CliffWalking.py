#!/usr/bin/env python
#
# Author: Marco Meschini
# Created: 08-09-2024
#

"""
Module Name: CliffWalking.py

This module implements Q-Learning algorithm to solve the Cliff Walking problem.
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

def main(): 
    env = gym.make('CliffWalking-v0')
    agent = QlearningAgent(
        env, discount_factor=1, exploration_rate=0.1, learning_rate=0.01)

    num_episodes = 30000
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
            done = terminated or truncated
            agent.update_Q(state, action, reward, next_state, done)
            
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
