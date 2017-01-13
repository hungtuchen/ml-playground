import numpy as np
import math

from policy_evaluation import policy_eval

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a (prob, next_state, reward, done) tuple.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: Lambda discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        # evalute how new policy performs
        V = policy_eval_fn(policy, env, discount_factor)
        # prepare for new policy, since new policy will be deterministic
        # we init probability for all actions as 0.0 and give only the best 1.0
        new_policy = np.zeros_like(policy)

        is_policy_optimized = True
        # iterate over state
        for s in range(env.nS):
            action_taken = policy[s].argmax()
            # value of current state given action, we will use it to choose best action
            action_values = np.zeros(env.nA)
            # iterate over action
            for a in range(env.nA):
                # iterate over next state given transition probability
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            # choose best action based on which action give us max value
            best_action = action_values.argmax()
            # if previous choosen action base on last policy does not equal to new action
            # based on max action value, then we didn't obtain optimal policy
            if action_taken != best_action:
                is_policy_optimized = False
            # update new policy no matter what
            new_policy[s][best_action] = 1.0
        if is_policy_optimized:
            return policy, V

        policy = new_policy
