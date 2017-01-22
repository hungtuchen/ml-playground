import numpy as np
from collections import defaultdict

def create_greedy_policy(Q, nA):
    """
    Creates a greedy policy based on Q values.

    Args:
        Q: A dictionary that maps from state -> action values
        nA: Number of actions in the environment.

    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities.
    """

    def policy_fn(observation):
        probs = np.zeros(nA)
        greedy_action = Q[observation].argmax()
        probs[greedy_action] = 1.0

        return probs

    return policy_fn

def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
    """
    Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
    Finds an optimal greedy policy.
    Ch5.7 Sutton & Barto, Reinforcement Learning: An Introduction
    Args:
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        behavior_policy: The behavior to follow while generating episodes.
            A function that given an observation returns a vector of probabilities for each action.
        discount_factor: Lambda discount factor.

    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities. This is the optimal greedy policy.
    """

    # The final action-value function.
    # A dictionary that maps state -> action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # A cumulative sum of weighted importance sampling ratio
    C = defaultdict(lambda: np.zeros(env.action_space.n))

    greedy_policy = create_greedy_policy(Q, env.action_space.n)

    for episode in range(num_episodes):
        # keep track of visited rewards, states, actions in current episode
        visited_rewards = []
        visited_state_actions = []

        current_state = env.reset()
        while True:
            # choose the action based on behavior_policy
            probs = behavior_policy(current_state)
            action = np.random.choice(np.arange(len(probs)), p=probs)

            next_state, reward, done, _ = env.step(action)
            visited_state_actions.append((current_state, action))
            visited_rewards.append(reward)

            if done:
                # cumulative reward
                G = 0
                # cumulative importance sampling ratio
                W = 1.0
                # iterate episode backwards and incrementally
                for t, state_action in reversed(list(enumerate(visited_state_actions))):
                    state, action = state_action
                    G = discount_factor * G + visited_rewards[t]
                    C[state][action] += W
                    # We use weighted importance sampling so divide by weighted average
                    Q[state][action] += (W / C[state][action]) * (G - Q[state][action])

                    # improve the policy based on new evaluated Q
                    greedy_policy = create_greedy_policy(Q, env.action_space.n)
                    # if target policy is not the same as behavior policy, then we can't
                    # evaluate it (not the same trajectory), so break
                    greedy_action = greedy_policy(state).argmax()
                    if action != greedy_action:
                        break
                    W *= 1.0 / behavior_policy(state)[action]
                # current episode is over
                break
            else:
                current_state = next_state


    return Q, greedy_policy
