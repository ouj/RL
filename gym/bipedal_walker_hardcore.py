#!/usr/bin/env python3

import gym
import tensorflow as tf
import os
import numpy as np
from datetime import datetime

def set_random_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)


def update_weights(weights, update, factor):
    new_weights = []
    for w1, w2 in zip(weights, update):
        new_weights.append(w1 + w2 * factor)
    return new_weights

def mutate_weights(weights, sigma):
    mutation = np.random.randn(len(weights))
    return weights + mutation * sigma

class EvoModel:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        observation_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0]

        initializer = tf.keras.initializers.random_normal(mean=0.0, stddev=10)
        inputs = tf.keras.layers.Input(shape=(observation_dim,))
        dense1 = tf.keras.layers.Dense(
            units=200,
            kernel_initializer=initializer,
            activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dense(
            units=action_dim,
            kernel_initializer=initializer,
            activation=tf.nn.sigmoid)(dense1)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    def get_1d_weights(self):
        return np.concatenate(
            [w.ravel() for w in self.model.get_weights()]
        )

    def set_1d_weights(self, weights):
        new_weights = []
        index = 0
        for w in self.model.get_weights():
            new_weights.append(weights[index:index+w.size].reshape(w.shape))
            index += w.size
        self.model.set_weights(new_weights)

    def predict(self, observation):
        action = self.model.predict(np.atleast_2d(observation))[0]
        action_min = self.action_space.low
        action_max = self.action_space.high
        action = (action_max - action_min) * action + action_min
        return np.clip(action, action_min, action_max)

    def mutate(self, sigma):
        new_model = EvoModel(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
        new_model.set_1d_weights(mutate_weights(self.get_1d_weights(), sigma))
        return new_model


def play_episode(env, model, render=False):
    observation = env.reset()
    done = False
    episode_return = 0
    episode_length = 0
    while not done:
        action = model.predict(observation)
        observation, reward, done, _ = env.step(action)
        episode_return += reward
        episode_length += 1
        if render:
            env.render()
    return episode_return, episode_length

def save_weight(weights, filename="evolution.npz"):
    np.savez(filename, weights)

def load_weight(filename="evolution.npz"):
    if not os.path.exists(filename):
        return None
    return np.load(filename)

def evolute(
    env,
    iterations=100,
    population_size=10,
    sigma=1.0,
    learning_rate=0.03,
    weights=None
):
    model = EvoModel(env.observation_space, env.action_space)
    if weights is None:
        weights = model.get_1d_weights()

    iteration_reward = np.zeros(iterations)
    for t in range(iterations):
        t0 = datetime.now()

        returns = np.zeros(population_size) # episode return
        mutations = np.random.randn(population_size, len(weights))

        for p in range(population_size):
            new_weights = weights + mutations[p] * sigma
            model.set_1d_weights(new_weights)
            episode_return, episode_length = play_episode(env, model, render=False)
            returns[p] = episode_return


        m = returns.mean()
        s = returns.std()
        if s == 0:
            continue

        iteration_reward[t] = m
        print ("Iteration reward:", m)
        A = (returns - m) / s

        weights = weights + learning_rate / (population_size * sigma) * np.dot(mutations.T, A)

        # update the learning rate
        learning_rate *= 0.992354
        sigma *= 0.999
        sigma = max(0.1, sigma)

        print("Iter:", t, "Avg Reward: %.3f" % m, "Max:", returns.max(), "Duration:", (datetime.now() - t0))

        if t != 0 and t % 10 == 0:
            model.set_1d_weights(weights)
            episode_return, episode_length = play_episode(env, model, render=True)
            save_weight(weights)

    return weights

def main():
    set_random_seed(0)
    env = gym.make("BipedalWalkerHardcore-v2")
    weights = load_weight()
    weights = evolute(
        env,
        iterations=1000,
        population_size=100,
        weights=weights
    )
    save_weight(weights)


if __name__ == "__main__":
    main()





