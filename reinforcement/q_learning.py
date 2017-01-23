import numpy as np
from collections import defaultdict
import itertools

from utils import plotting
from utils.policy import make_epsilon_greedy_policy

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        current_state = env.reset()
        # keep track number of time-step per episode only for plotting
        for t in itertools.count():
            # choose the action based on epsilon greedy policy
            action_probs = policy(current_state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # not the action which agent will actually follow
            greedy_next_action = Q[next_state].argmax()
            # evaluate Q using estimated action value of (next_state, greedy_next_action)
            td_target = reward + discount_factor * Q[next_state][greedy_next_action]
            td_error = td_target - Q[current_state][action]
            Q[current_state][action] += alpha * td_error

            # improve epsilon greedy policy using new evaluate Q
            policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if done:
                break
            else:
                current_state = next_state

    return Q, stats
