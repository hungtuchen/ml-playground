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

    while True:
        V_converged = True
        # iterate over state
        for s in range(env.nS):
            new_v = 0
            # iterate over action
            for a in range(env.nA):
                # iterate over next state given transition probability
                for prob, next_state, reward, done in env.P[s][a]:
                    # Bellman expectation backup
                    new_v += policy[s][a] * prob * (reward + discount_factor * V[next_state])
            # if the update between two iterration is less then theta
            # for every state of V, then the value function is converged
            # other is not
            if np.abs(new_v - V[s]) > theta:
                V_converged = False
            # update new value
            V[s] = new_v
        # stop if V_converged
        if V_converged:
            return np.array(V)
