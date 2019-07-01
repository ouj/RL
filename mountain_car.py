#!/usr/bin/env python3

import gym
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
import tensorflow as tf


# Inspired by https://github.com/dennybritz/reinforcement-learning
class FeatureTransformer:
    def __init__(self, env, n_components=500):
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        # Used to converte a state to a featurizes represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
            ])
        example_features = featurizer.fit_transform(scaler.transform(observation_examples))

        self.dimensions = example_features.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        # print "observations:", observations
        scaled = self.scaler.transform(observations)
        # assert(len(scaled.shape) == 2)
        return self.featurizer.transform(scaled)


class Regressor:
    def __init__(self, learning_rate = 0.1):
        self.X = tf.placeholder(tf.float32, shape=(None, 2000))
        self.Y = tf.placeholder(tf.float32, shape=(None,))

        self.W = tf.Variable(tf.random_normal(shape=(2000, 1)))

        Y_hat = tf.reshape(tf.matmul(self.X, self.W), [-1])
        delta = self.Y - Y_hat

        gamma = 0.001
        cost = tf.reduce_sum(delta * delta) + gamma * tf.reduce_sum(self.W * self.W)
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        self.predict_op = Y_hat

        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def fit(self, X, Y):
        self.session.run(self.train_op, feed_dict = {self.X: X, self.Y: Y})

    def predict(self, X):
        return self.session.run(self.predict_op, feed_dict = {self.X: X})

class Model:

    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            model = Regressor()
            self.models.append(model)

    def predict(self, s):
        X = self.feature_transformer.transform([s])
        result = np.stack([m.predict(X) for m in self.models]).T
        assert(len(result.shape) == 2)
        return result

    def update(self, s, a, G):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape) == 2)
        self.models[a].fit(X, [G])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))


# returns a list of states_and_rewards, and the total reward
def play_one(model, env, eps, gamma, render=False):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, _ = env.step(action)

        # update the model
        next = model.predict(observation)
        # assert(next.shape == (1, env.action_space.n))
        G = reward + gamma * np.max(next[0])
        model.update(prev_observation, action, G)

        totalreward += reward
        iters += 1
        if render:
            env.render()

    return totalreward


def plot_cost_to_go(env, estimator, num_tiles=20):
     x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
     y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
     X, Y = np.meshgrid(x, y)
     # both X and Y will be of shape (num_tiles, num_tiles)
     Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
     # Z will also be of shape (num_tiles, num_tiles)

     fig = plt.figure(figsize=(10, 5))
     ax = fig.add_subplot(111, projection='3d')
     surf = ax.plot_surface(
         X, Y, Z,
         rstride=1, cstride=1,
         cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
     ax.set_xlabel('Position')
     ax.set_ylabel('Velocity')
     ax.set_zlabel('Cost-To-Go == -V(s)')
     ax.set_title("Cost-To-Go Function")
     fig.colorbar(surf)
     plt.show()


def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


def main(show_plots=True):
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft, "constant")
    gamma = 0.99

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)


    N = 300
    totalrewards = np.empty(N)
    for n in range(N):
        # eps = 1.0/(0.1*n+1)
        eps = 0.1*(0.97**n)
        # eps = 1.0/np.sqrt(n+1)
        totalreward = play_one(model, env, eps, gamma, render=False)
        totalrewards[n] = totalreward
        print("episode:", n, "total reward:", totalreward)
        if n != 0 and n % 100 == 0:
            print("epsilon:", eps)
            print("avg reward for last 100 episodes:", totalrewards[n-100:n].mean())
    print("total steps:", -totalrewards.sum())

    if show_plots:
        plt.plot(totalrewards)
        plt.title("Rewards")
        plt.show()

        plot_running_avg(totalrewards)

        # plot the optimal state-value function
        plot_cost_to_go(env, model)


    play_one(model, env, 0, 0, render=True)
    env.close()


if __name__ == '__main__':
  # for i in range(10):
  #   main(show_plots=False)
  main()