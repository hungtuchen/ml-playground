import numpy as np
from collections import defaultdict
import itertools
import random

from utils import plotting

def make_double_q_epsilon_greedy_policy(epsilon, nA, Q1, Q2):
    """
    Creates an epsilon-greedy policy based on sum of given Q-functions and epsilon.

    Args:
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
        Q1, Q2: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)

    Returns:
        A function that takes the state as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(state):
        # 1 / epsilon for non-greedy actions
        probs = (epsilon / nA) * np.ones(nA)

        summed_Q = Q1[state] + Q2[state]

        greedy_action = summed_Q.argmax()
        # (1 / epsilon + (1 - epsilon)) for greedy action
        probs[greedy_action] += 1.0 - epsilon

        return probs

    return policy_fn

def double_q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Double Q-Learning algorithm: Off-policy TD control that avoid maxmization.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q1, Q2, episode_lengths).
        Q1 + Q2 is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value functions.
    # A nested dictionary that maps state -> (action -> action-value).
    Q1 = defaultdict(lambda: np.zeros(env.action_space.n))
    Q2 = defaultdict(lambda: np.zeros(env.action_space.n))

    # keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    policy = make_double_q_epsilon_greedy_policy(epsilon, env.action_space.n, Q1, Q2)

    for i_episode in range(num_episodes):
        current_state = env.reset()
        # keep track number of time-step per episode only for plotting
        for t in itertools.count():
            # choose the action based on epsilon greedy policy
            action_probs = policy(current_state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            if random.random() < 0.5:
                # Update Q1: using Q1 to select max action yet using Q2's estimate.
                greedy_next_action = Q1[next_state].argmax()
                td_target = reward + discount_factor * Q2[next_state][greedy_next_action]
                td_error = td_target - Q1[current_state][action]
                Q1[current_state][action] += alpha * td_error
            else:
                # Update Q2: using Q2 to select max action yet using Q1's estimate.
                greedy_next_action = Q2[next_state].argmax()
                td_target = reward + discount_factor * Q1[next_state][greedy_next_action]
                td_error = td_target - Q2[current_state][action]
                Q2[current_state][action] += alpha * td_error

            # improve epsilon greedy policy using new evaluate Q
            policy = make_double_q_epsilon_greedy_policy(epsilon, env.action_space.n, Q1, Q2)

            # update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if done:
                break
            else:
                current_state = next_state

    return Q1, Q2, stats
