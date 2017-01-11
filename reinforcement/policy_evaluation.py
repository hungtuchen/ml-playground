import numpy as np
def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a (prob, next_state, reward, done) tuple.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    # just prevent while loop never stops
    last_V = np.ones_like(V)

    # stop when changes of non-terminal state less then theta
    while np.all(np.abs(last_V[1:-1] - V[1:-1]) > theta):
        last_V = np.copy(V)
        # iterate over state
        for s in range(env.nS):
            current_v = 0
            # iterate over action
            for a in range(env.nA):
                prob, next_state, reward, done = env.P[s][a][0]
                # Bellman expectation backup
                # If you replace last_V with V, then it's In-Place Dynamic Programming
                current_v += policy[s][a] * (reward + discount_factor * np.sum(prob * last_V[next_state]))
            # update new value
            V[s] = current_v

    return np.array(V)
