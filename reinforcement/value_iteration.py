import numpy as np

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment. env.P represents the transition probabilities of the environment.
        theta: Stopping threshold. If the value of all states changes less than theta
            in one iteration we are done.
        discount_factor: lambda time discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """


    V = np.zeros(env.nS)

    while True:
        V_converged = True
        # init a empty policy every time, we can choose the best policy along with
        # find optimal action value
        policy = np.zeros([env.nS, env.nA])
        for s in range(env.nS):
            action_values = np.zeros(env.nA)
            # use one-step lookahead and update v to best action value
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            max_v = action_values.max()

            # converged only when V reach optimal (max action value doesn't change anymore)
            if np.abs(max_v - V[s]) > theta:
                V_converged = False
            # update v and corresponding policy
            V[s] = max_v
            policy[s][action_values.argmax()] = 1.0

        if V_converged:
            return policy, V
