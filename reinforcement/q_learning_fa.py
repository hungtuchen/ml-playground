import numpy as np
import itertools
from sklearn.linear_model import SGDRegressor

from utils import plotting

class Estimator():
    """
    Value Function approximator.
    """

    def __init__(self, env, scaler, featurizer):
        self.env = env
        self.scaler = scaler
        self.featurizer = featurizer
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        for _ in range(self.env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            initial_feature = self.featurize_state(env.reset()).reshape(1, -1)
            model.partial_fit(initial_feature, [0])
            self.models.append(model)

    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]

    def predict(self, s, a=None):
        """
        Makes value function predictions.

        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for

        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.

        """
        models = self.models
        feature = self.featurize_state(s).reshape(1, -1)

        if a is not None:
            return models[a].predict(feature)[0]
        else:
            return [model.predict(feature) for model in models]

    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        feature = self.featurize_state(s).reshape(1, -1)
        model = self.models[a]
        model.partial_fit(feature, [y])

def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning_fa(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):
        policy = make_epsilon_greedy_policy(
            estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)

        current_state = env.reset()
        # keep track number of time-step per episode only for plotting
        for t in itertools.count():
            # choose the action based on epsilon greedy policy
            action_probs = policy(current_state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # use the greedy action to evaluate Q, not the one we actually follow
            greedy_next_action = np.argmax(estimator.predict(next_state))
            # evaluate Q using estimated action value of (next_state, greedy_next_action)
            td_target = reward + discount_factor * estimator.predict(next_state, greedy_next_action)
            # update weights
            estimator.update(current_state, action, td_target)

            # update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if done:
                break
            else:
                current_state = next_state

    return stats
