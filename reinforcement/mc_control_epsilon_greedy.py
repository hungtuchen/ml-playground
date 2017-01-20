import numpy as np
from collections import defaultdict

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the state as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(state):
        action_values = Q[state]
        greedy_action = action_values.argmax()
        # 1 / epsilon for non-greedy actions, (1 / epsilon + (1 - epsilon)) for greedy action
        probs = (epsilon / nA) * np.ones(nA)
        probs[greedy_action] += 1.0 - epsilon

        return probs

    return policy_fn

def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.

    Args:
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an state as an argument and returns
        action probabilities
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final action-value function.
    # A nested dictionary that maps state -> [...action_value](key corresponds to action)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Initialize the epsilon-greedy policy
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for episode in range(num_episodes):
        # keep track of visited rewards, states, actions in current episode
        visited_rewards = []
        visited_state_actions = []

        current_state = env.reset()
        while True:
            # choose the action based on policy
            probs = policy(current_state)
            action = np.random.choice(np.arange(len(probs)), p=probs)

            next_state, reward, done, _ = env.step(action)
            visited_state_actions.append((current_state, action))
            visited_rewards.append(reward)

            if done:
                # we only take the first time the (state, action) is visited in the episode
                # into account if we use first-visit Monte Carlo methonds
                was_state_action_visited = {}
                for i, state_action in enumerate(visited_state_actions):
                    # uncomment this part if you want to use first-visit
                    # if state_action not in was_state_action_visited:
                        # was_state_action_visited[state_action] = True
                    returns_count[state_action] += 1.0
                    # calculate the return (expected rewards from current state_action onwards)
                    # Note: we need to take care of discount_factor
                    return_ = 0.0
                    for j, reward in enumerate(visited_rewards[i:]):
                        return_ += (discount_factor ** j) * reward
                    returns_sum[state_action] += return_

                # evaluate Q after every epsilon (We can improve policy when we have more experience)
                for state_action, count in returns_count.items():
                    state, action = state_action
                    Q[state][action] = returns_sum[state_action] / count

                # improve the policy based on new evaluated Q
                policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
                break
            else:
                current_state = next_state

    return Q, policy
