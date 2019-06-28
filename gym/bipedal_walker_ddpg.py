#!/usr/bin/env python3

import os

import gym
import numpy as np
import tensorflow as tf
from rl.helpers import set_random_seed
from rl.mlp import MLPNetwork


# Environment
env = gym.make("BipedalWalker-v2")

# Configurations
LEARNING_RATE = 1e-3

set_random_seed(0)


class MuNetwork(MLPNetwork):
    def __init__(self, output_dim, activation=tf.nn.relu, trainable=True):
        super(MLPNetwork, self).__init__()
        self.layers = [
            tf.layers.Dense(
                units=512,
                activation=activation,
                trainable=trainable,
                name="W",
                kernel_initializer=tf.initializers.glorot_normal,
            ),
            tf.layers.Dense(
                units=512,
                activation=activation,
                trainable=trainable,
                name="V",
                kernel_initializer=tf.initializers.glorot_normal,
            ),
            tf.layers.Dense(
                units=output_dim,
                trainable=trainable,
                name="MU",
                kernel_initializer=tf.initializers.glorot_normal,
            ),
        ]


class QNetwork(tf.layers.Layer):
    def __init__(self, activation=tf.nn.relu, trainable=True):
        super(MuNetwork, self).__init__()
        self.layers = [
            tf.layers.Dense(
                units=512,
                activation=activation,
                trainable=trainable,
                name="W",
                kernel_initializer=tf.initializers.glorot_normal,
            ),
            tf.layers.Dense(
                units=512,
                activation=activation,
                trainable=trainable,
                name="V",
                kernel_initializer=tf.initializers.glorot_normal,
            ),
            tf.layers.Dense(
                units=1,
                trainable=trainable,
                name="Q",
                kernel_initializer=tf.initializers.glorot_normal,
            ),
        ]


class ReplayBuffer:
    def __init__(self, observation_dim, action_dim, max_size=1000000):
        self.index = 0
        self.size = 0
        self.max_size = max_size
        self.states = np.zeros([max_size], dtype=np.float32)
        self.next_states = np.zeros([max_size, observation_dim], dtype=np.float32)
        self.actions = np.zeros([max_size, action_dim], dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)

    def number_of_samples(self):
        return self.size

    def store(self, observation, action, reward, next_observation, done):
        self.states[self.index] = observation
        self.next_states[self.index] = next_observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        assert batch_size <= self.size
        indexes = np.random.randint(0, self.size, size=batch_size)
        return dict(
            s=self.states[indexes],
            s2=self.next_states[indexes],
            a=self.actions[indexes],
            r=self.rewards[indexes],
            d=self.dones[indexes],
        )


replay_buffer = ReplayBuffer(
    observation_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0]
)
