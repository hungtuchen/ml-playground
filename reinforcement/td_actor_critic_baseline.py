import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

from utils import plotting

class PolicyEstimator(nn.Module):
    """
    Policy Function approximator.
    """

    def __init__(self, D_in, D_out,  hidden_size = 128):
        super(PolicyEstimator, self).__init__()
        # define network structure
        self.W1 = nn.Linear(D_in, hidden_size)
        self.W2 = nn.Linear(hidden_size, D_out)

    def forward(self, state):
        h = F.relu(self.W1(state))
        action_scores = self.W2(h)
        return F.softmax(action_scores)

class ValueEstimator(nn.Module):
    """
    Value Function approximator.
    """

    def __init__(self, D_in, hidden_size = 128):
        super(ValueEstimator, self).__init__()
        # define network structure
        self.W1 = nn.Linear(D_in, hidden_size)
        # output a score
        self.W2 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        h = F.relu(self.W1(state))
        state_values = self.W2(h)
        return state_values

def td_actor_critic_baseline(env, policy_estimator, policy_optimizer, value_estimator, value_optimizer,
                       num_episodes, discount_factor=1.0, render=True):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm with Baseline.
    Optimizes the policy function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        policy_estimator: Policy Function to be optimized
        policy_optimizer: Optimizer for Policy Function
        value_estimator: Value function approximator, used as a baseline
        value_optimizer: Optimizer for Value Function
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
        render: Render the training process or not

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    running_reward = 0
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):
        episode_rewards = []

        state = env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0)
        for t in count(1):
            # Calculate the probability distribution of actions
            probs = policy_estimator(Variable(state))
            # Select action by distribution estimated above
            action = probs.multinomial()

            next_state, reward, done, _ = env.step(action.data[0, 0])
            next_state = torch.from_numpy(next_state).float().unsqueeze(0)
            if render:
                env.render()
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            episode_rewards.append(reward)

            # Calculate TD(0) target
            td_target = reward + discount_factor * value_estimator(Variable(next_state, requires_grad=False))
            # Calculate estimated state value as baseline
            baseline = value_estimator(Variable(state))
            # Calculate TD(0) error
            td_error = td_target - baseline

            # Registers a reward obtained as a result of a stochastic process.
            # Differentiating stochastic nodes requires providing them with reward value.
            action.reinforce(td_error.data)

            # Define creterion and calculate loss for value funcion
            value_loss = F.smooth_l1_loss(baseline, td_target)

            # Remove gradient from previous steps
            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()

            # Perform backward pass
            action.backward()
            value_loss.backward()

            # Use optimizer to update
            policy_optimizer.step()
            value_optimizer.step()

            if done:
                break
            else:
                state = next_state

        # Book-keep the running reward
        running_reward = running_reward * 0.99 + sum(episode_rewards) * 0.01
        if i_episode % 10 == 0:
            print('Episode {}\tRunning reward: {:.2f}'.format(i_episode, running_reward))
        if running_reward > 200:
            print("Solved! Running reward is now {} and " \
                "the last episode runs to {} time steps!".format(running_reward, t))
            break

    return stats
