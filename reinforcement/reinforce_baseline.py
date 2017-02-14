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

def select_action_baseline(state, policy_estimator, value_estimator):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy_estimator(Variable(state))
    action = probs.multinomial()
    baseline = value_estimator(Variable(state))
    return action, baseline

def discount_rewards(rewards, gamma):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_rewards = np.zeros_like(rewards)
  running_add = 0
  for t in reversed(range(len(rewards))):
    running_add = running_add * gamma + rewards[t]
    discounted_rewards[t] = running_add
  return discounted_rewards

def reinforce_baseline(env, policy_estimator, policy_optimizer, value_estimator, value_optimizer,
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
        episode_baselines = []
        episode_actions = []
        episode_rewards = []

        state = env.reset()
        for t in count(1):
            action, baseline = select_action_baseline(state, policy_estimator, value_estimator)
            state, reward, done, _ = env.step(action.data[0, 0])
            if render:
                env.render()
            # keep track of visited action, reward and baseline for later update
            episode_baselines.append(baseline)
            episode_actions.append(action)
            episode_rewards.append(reward)
            # update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if done:
                break

        # start updating policy and value estimator
        discount_rs = discount_rewards(episode_rewards, discount_factor)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discount_rs -= discount_rs.mean()
        discount_rs /= discount_rs.std()

        # define creterion and calculate loss for value funcion
        value_criterion = nn.MSELoss()
        value_loss = 0.0

        # Registers a reward obtained as a result of a stochastic process.
        # Differentiating stochastic nodes requires providing them with reward value.
        for baseline, action, r in zip(episode_baselines, episode_actions, discount_rs):
            action.reinforce(r - baseline.data)
            value_loss += value_criterion(baseline, Variable(torch.Tensor([r]), requires_grad=False))

        # remove gradient from previous steps
        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()

        # backward function additionaly requires specifying grad_variables.
        # It should be a sequence of matching length, that containins gradient of the
        # differentiated function w.r.t. corresponding variables
        # (None is an acceptable value for all variables that don’t need gradient tensors).
        autograd.backward(episode_actions, [None for _ in episode_actions])
        value_loss.backward()

        # use optimizer to update
        policy_optimizer.step()
        value_optimizer.step()

        # book-keep the running reward
        running_reward = running_reward * 0.99 + sum(episode_rewards) * 0.01
        if i_episode % 10 == 0:
            print('Episode {}\tRunning reward: {:.2f}'.format(i_episode, running_reward))
        if running_reward > 200:
            print("Solved! Running reward is now {} and " \
                "the last episode runs to {} time steps!".format(running_reward, t))
            break

    return stats
