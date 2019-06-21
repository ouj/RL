#!/usr/bin/env python3.7
import os
import sys
from datetime import datetime

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from rl.ann import get_vars
from rl.helpers import atleast_4d, set_random_seed
from rl.stacked_replay_buffer import StackedFrameReplayBuffer


# Set random seeds
set_random_seed(0)

# Path and folders
FILENAME = "space_invadors_dpn"
TS = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
MONITOR_DIR = os.path.join("video", FILENAME + "_" + TS)
LOGGING_DIR = os.path.join("log", FILENAME + "_" + TS)

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
GAMMA = 0.99
DECAY = 0.99
MINIMAL_SAMPLES = 1000
ITERATIONS = 20000
DEMO_NUMBER = 10
STACK_SIZE = 1

env = gym.make("SpaceInvaders-v4")

# Image preprocessing
class ImagePreprocessor:
    def __init__(self, height, width, channels):
        with tf.variable_scope("image_preprocessor"):
            self.input = tf.placeholder(shape=[height, width, channels], dtype=tf.uint8)
            t = tf.image.convert_image_dtype(self.input, dtype=tf.float32)
            t = tf.image.rgb_to_grayscale(t)
            self.output = tf.squeeze(t)

    def transform(self, frame, session=None):
        session = session if session is not None else tf.get_default_session()
        return session.run(self.output, feed_dict={self.input: frame})


# Layer Definitions
class ConvLayer(tf.layers.Layer):
    def __init__(self, activation=tf.nn.elu):
        super(ConvLayer, self).__init__()
        with tf.variable_scope("conv"):
            self.conv1 = tf.layers.Conv2D(
                filters=32,
                kernel_size=8,
                strides=4,
                padding="valid",
                activation=activation,
                kernel_initializer=tf.initializers.glorot_normal,
                name="conv1",
            )
            self.conv2 = tf.layers.Conv2D(
                filters=64,
                kernel_size=4,
                strides=2,
                padding="valid",
                activation=activation,
                kernel_initializer=tf.initializers.glorot_normal,
                name="conv2",
            )
            self.conv3 = tf.layers.Conv2D(
                filters=64,
                kernel_size=3,
                strides=2,
                padding="valid",
                activation=activation,
                kernel_initializer=tf.initializers.glorot_normal,
                name="conv3",
            )
            self.flatten = tf.layers.Flatten(name="flatten")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.flatten(x)


class QLayer(tf.layers.Layer):
    def __init__(self, output_dim, scope, activation=tf.nn.relu, trainable=True):
        super(QLayer, self).__init__()
        with tf.variable_scope(scope):
            self.W = tf.layers.Dense(
                units=512, activation=activation, trainable=trainable, name="W"
            )
            self.Q = tf.layers.Dense(
                units=output_dim, trainable=trainable, name="Q"
            )

    def call(self, inputs):
        x = self.W(inputs)
        return self.Q(x)


tf.reset_default_graph()

# Inputs
X = tf.placeholder(
    shape=(
        None,
        env.observation_space.shape[0],
        env.observation_space.shape[1],
        STACK_SIZE,
    ),
    dtype=tf.float32,
    name="x",
)

X2 = tf.placeholder(
    shape=(
        None,
        env.observation_space.shape[0],
        env.observation_space.shape[1],
        STACK_SIZE,
    ),
    dtype=tf.float32,
    name="x2",
)

# Convolution
conv_layer = ConvLayer()
Z = conv_layer(X)
Z2 = conv_layer(X2)

# Deep Q Network

Q = QLayer(output_dim=env.action_space.n, scope="main", trainable=True)(Z)
Q2 = QLayer(output_dim=env.action_space.n, scope="target", trainable=False)(Z2)

R = tf.placeholder(dtype=tf.float32, shape=(None,))  # reward
D = tf.placeholder(dtype=tf.float32, shape=(None,))  # done
A = tf.placeholder(dtype=tf.int32, shape=(None,))  # action_dim

selected_q = tf.reduce_sum(Q * tf.one_hot(A, env.action_space.n), reduction_indices=[1])
next_q = tf.math.reduce_max(q2, axis=1)
g = tf.stop_gradient(R + GAMMA * next_q * (1 - D))
q_loss = tf.reduce_sum(tf.square(selected_q - g))
train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(q_loss)

# Copy main network params to target networks
target_init = tf.group(
    [
        tf.assign(v_targ, v_main)
        for v_main, v_targ in zip(get_vars("main"), get_vars("target"))
    ]
)

# Use soft updates to update the target networks
target_update = tf.group(
    [
        tf.assign(v_targ, DECAY * v_targ + (1 - DECAY) * v_main)
        for v_main, v_targ in zip(get_vars("main"), get_vars("target"))
    ]
)

image_preprocessor = ImagePreprocessor(*env.observation_space.shape)

session = tf.Session()
session.run(tf.global_variables_initializer())
session.run(target_init)

writer = tf.summary.FileWriter(LOGGING_DIR)
writer.add_graph(session.graph)

#%% Replay Buffer

replay_buffer = StackedFrameReplayBuffer(
    frame_width=env.observation_space.shape[0],
    frame_height=env.observation_space.shape[1],
    channels=env.observation_space.shape[2],
    stack_size=STACK_SIZE,
    action_dim=1,
)


# %% Play


def update_state(state, observation):
    return np.concatenate((state[:, :, STACK_SIZE - 1 :], observation), axis=2)


def sample_action(env, state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        feed_dict = {x: atleast_4d(state)}
        q_s_a = session.run(q, feed_dict)[0]
        return np.argmax(q_s_a)


def play_once(env, epsilon, render=False):
    observation = env.reset()
    done = False
    steps = 0
    total_return = 0
    state = np.concatenate([observation] * 4, axis=2)
    while not done:
        action = sample_action(env, state, epsilon)
        observation, reward, done, _ = env.step(action)

        next_state = update_state(state, observation)

        replay_buffer.store(observation, action, reward, done)

        steps += 1
        state = next_state
        total_return += reward
        if render:
            env.render()
    return steps, total_return


#%% Train
def train(steps):
    losses = np.zeros(steps)
    for n in range(steps):
        batch = replay_buffer.sample_batch(BATCH_SIZE)

        feed_dict = {
            x: batch["s"],
            x2: batch["s2"],
            a: np.squeeze(batch["a"]),
            r: batch["r"],
            d: batch["d"],
        }
        session.run(train_op, feed_dict)
        session.run(target_update)
        losses[n] = session.run(q_loss, feed_dict)
    return np.mean(losses)


# %% main loop
returns = []

for n in range(ITERATIONS):
    epsilon = 1 / np.sqrt(n + 1)
    steps, total_return = play_once(env, epsilon, render=False)

    returns.append(total_return)
    if MINIMAL_SAMPLES < replay_buffer.number_of_samples():
        loss = train(steps)
        print("Trained for %d steps, mean q loss %f" % (steps, loss))

    if n != 0 and n % 10 == 0:
        print(
            "Episode:",
            n,
            "Average Returns:",
            np.mean(returns[n - 10 :]),
            "epsilon:",
            epsilon,
        )

#%% Demo

env = gym.wrappers.Monitor(env, MONITOR_DIR)
for n in range(DEMO_NUMBER):
    play_once(env, 0.0, render=True)

# %%Close Environment
env.close()

#%% Report

plt.figure()
plt.plot(returns)
plt.title("Returns")
plt.savefig(os.path.join(monitor_dir, "returns.pdf"))
