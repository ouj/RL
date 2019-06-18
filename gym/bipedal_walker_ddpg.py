#!/usr/bin/env python3

import os
import gym
import numpy as np
import tensorflow as tf
from rl.replay_buffer import ReplayBuffer

# Configurations
LEARNING_RATE = 1e-3

# Set random seeds
def set_random_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
set_random_seed(0)


# Setup
env = gym.make("BipedalWalker-v2")
replay_buffer = ReplayBuffer(
    observation_shape=env.observation_space.shape,
    action_shape=env.action_space.shape
)
