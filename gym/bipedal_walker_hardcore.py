#!/usr/bin/env python3

import gym
import tensorflow as tf
import numpy as np
from rl.replay_buffer import ReplayBuffer

def set_random_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)

# We will be solving this problem using evolution strategies
#
set_random_seed(0)
#env = gym.make("BipedalWalkerHardcore-v2")
#
#observation = env.reset()
#
#tf.reset_default_graph()
#
#observation_dim = env.observation_space.shape[0]
#action_dim = env.action_space.shape[0]
#
#tf.all_variables()
#
#inputs = tf.keras.layers.Input(shape=(observation_dim,))
#dense1 = tf.keras.layers.Dense(
#    units=200, activation=tf.nn.relu)(inputs)
#outputs = tf.keras.layers.Dense(
#    units=action_dim, activation=tf.nn.sigmoid)(dense1)
#model1 = tf.keras.Model(inputs=inputs, outputs=outputs)
#model1.predict(np.atleast_2d(observation))
#
#weights = model1.get_weights()
#model1.set_weights(model1.get_weights())
#w = weights[0]
#noise = np.random.randn(*w.shape)
#sigma = 0.001
#w = w + sigma * noise

def mutate_weights(weights, sigma):
    new_weights = []
    for w in weights:
        noise = np.random.randn(*w.shape)
        w = w + sigma * noise
        new_weights.append(w)
    return new_weights

class EvoModel:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        observation_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0]

        inputs = tf.keras.layers.Input(shape=(observation_dim,))
        dense1 = tf.keras.layers.Dense(
            units=200, activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dense(
            units=action_dim, activation=tf.nn.sigmoid)(dense1)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

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
        new_model.set_weights(mutate_weights(self.get_weights(), sigma))
        return new_model

model = EvoModel(env.observation_space, env.action_space)
model.predict(observation)
model = model.mutate(0.001)

def play_episode(env, model, render=False):
    observation = env.reset()
    done = False
    episode_return = 0
    episode_length = 0
    while not done:
        action = model.predict(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_return += reward
        episode_length += 1
        if render:
            env.render()
    return episode_return, episode_length

play_episode(env, model, render=True)


