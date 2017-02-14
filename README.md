# ml-playground

This project reuses code and is highly inspired by (selected) great resources:

1. [MLAlgorithms](https://github.com/rushter/MLAlgorithms)
2. [reinforcement-learning](https://github.com/dennybritz/reinforcement-learning).

It's good to learn machine learning by implementing algorithms, and better if we can write them in an elegant and readable way. Then we can review later or others might easily learn from them.

So I manage to use succinct codes with demo example to illustrate the essence of ml algorithms.

## Supervised

- Linear Regression [[code]](supervised/linear_regression.py) [[demo]](supervised/linear_regression.ipynb)

- Logistic Regression [[code]](supervised/logistic_regression.py) [[demo]](supervised/logistic_regression.ipynb)

- Naive Bayes [[code]](supervised/naive_bayes.py) [[demo]](supervised/naive_bayes.ipynb)

- AdaBoost [[code]](supervised/adaboost.py) [[demo]](supervised/adaboost.ipynb)

## Unsupervised

- Kmeans [[code]](unsupervised/kmeans.py) [[demo]](unsupervised/kmeans.ipynb)

- PCA [[code]](unsupervised/pca.py) [[demo]](unsupervised/pca.ipynb)

## Reinforcement

- Dynamic Programming

  - Policy Evaluation [[code]](reinforcement/policy_evaluation.py) [[demo]](reinforcement/policy_evaluation.ipynb)

  - Policy Iteration [[code]](reinforcement/policy_iteration.py) [[demo]](reinforcement/policy_iteration.ipynb)

  - Value Iteration [[code]](reinforcement/value_iteration.py) [[demo]](reinforcement/value_iteration.ipynb)

- Monte Carlo Methods

  - Monte Carlo Prediction [[code]](reinforcement/mc_prediction.py) [[demo]](reinforcement/mc_prediction.ipynb)

  - Monte Carlo Control (with Epsilon-Greedy Policy) [[code]](reinforcement/mc_control_epsilon_greedy.py) [[demo]](reinforcement/mc_control_epsilon_greedy.ipynb)

  - Monte Carlo Off-Policy Control (with Weighted Importance Sampling) [[code]](reinforcement/mc_control_importance_sampling.py) [[demo]](reinforcement/mc_control_importance_sampling.ipynb)

- Temporal Difference Methods

  - SARSA [[code]](reinforcement/sarsa.py) [[demo]](reinforcement/sarsa.ipynb)
  - Q-Learning [[code]](reinforcement/q_learning.py) [[demo]](reinforcement/q_learning.ipynb)

- Value Function Approximation

  - Q-Learning (with Function Approximation) [[code]](reinforcement/q_learning_fa.py) [[demo]](reinforcement/q_learning_fa.ipynb)

- Policy Gradient Methods

  - REINFORCE with Baseline [[code]](reinforcement/reinforce_baseline.py) [[demo]](reinforcement/reinforce_baseline.ipynb)
