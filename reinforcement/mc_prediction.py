import numpy as np
from collections import defaultdict

def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.

    Args:
        policy: A function that maps an state to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final value function
    V = defaultdict(float)

    for episode in range(num_episodes):
        # keep track of visited rewards and states in current episode
        visited_rewards = []
        visited_states = []

        current_state = env.reset()
        while True:
            # choose the action based on policy
            probs = policy(current_state)
            action = np.random.choice(np.arange(len(probs)), p=probs)

            next_state, reward, done, _ = env.step(action)
            visited_states.append(current_state)
            visited_rewards.append(reward)

            if done:
                # we only take the first time the state is visited in the episode
                # into account if we use first-visit Monte Carlo methonds
                was_state_visited = {}
                for i, state in enumerate(visited_states):
                    # uncomment this part if you want to use first-visit
                    # if state not in was_state_visited:
                        # was_state_visited[state] = True
                    returns_count[state] += 1.0
                    # calculate the return (expected rewards from current state onwards)
                    # Note: we need to take care of discount_factor
                    return_ = 0.0
                    for j, reward in enumerate(visited_rewards[i:]):
                        return_ += (discount_factor ** j) * reward
                    returns_sum[state] += return_

                break
            else:
                current_state = next_state
    # use eperical mean to predict value function
    for state, count in returns_count.items():
        V[state] = returns_sum[state] / count

    return V
