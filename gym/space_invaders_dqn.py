#!/usr/bin/env python3.7
import itertools
import os
import sys
from collections import deque
from datetime import datetime

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from rl.helpers import atleast_4d, set_random_seed
from wrappers.atari_wrappers import EpisodicLifeEnv


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set random seeds
set_random_seed(0)

# Path and folders
FILENAME = "space_invadors_dpn"
TS = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
MONITOR_DIR = os.path.join("video", FILENAME + "_" + TS)
LOGGING_DIR = os.path.join("log", FILENAME)

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
GAMMA = 0.99
DECAY = 0.99
MINIMAL_SAMPLES = 1000
ITERATIONS = 20000
DEMO_NUMBER = 10
STACK_SIZE = 4

real_env = gym.make("SpaceInvaders-v4")
env = EpisodicLifeEnv(real_env)
test_env = gym.wrappers.Monitor(env, MONITOR_DIR)

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
        with tf.variable_scope("conv"):
            x = self.conv1(inputs)
            x = self.conv2(x)
            x = self.conv3(x)
            return self.flatten(x)


class QLayer(tf.layers.Layer):
    def __init__(self, output_dim, scope, activation=tf.nn.relu, trainable=True):
        super(QLayer, self).__init__()
        self.scope = scope
        self.W = tf.layers.Dense(
            units=512, activation=activation, trainable=trainable, name="W"
        )
        self.Q = tf.layers.Dense(units=output_dim, trainable=trainable, name="Q")

    def collect_variables(self):
        variables = []
        for layer in [self.W, self.Q]:
            variables += layer.variables
        return variables

    def copy_from(self, other_qlayer, session=None):
        assert isinstance(other_qlayer, QLayer)
        target_variables = self.collect_variables()
        source_variables = other_qlayer.collect_variables()
        copy_op = tf.group(
            [
                tf.assign(v_tgt, v_src)
                for v_tgt, v_src in zip(target_variables, source_variables)
            ]
        )
        session = session if session is not None else tf.get_default_session()
        session.run(copy_op)

    def update_from(self, other_qlayer, decay, session=None):
        target_variables = self.collect_variables()
        source_variables = other_qlayer.collect_variables()
        update_op = tf.group(
            [
                tf.assign(v_tgt, decay * v_tgt + (1 - decay) * v_src)
                for v_tgt, v_src in zip(target_variables, source_variables)
            ]
        )
        session = session if session is not None else tf.get_default_session()
        session.run(update_op)

    def is_equal(self, other_qlayer, session=None):
        assert isinstance(other_qlayer, QLayer)
        target_variables = self.collect_variables()
        source_variables = other_qlayer.collect_variables()
        equal_op = tf.reduce_all(
            [
                tf.reduce_all(tf.equal(v_tgt, v_src))
                for v_tgt, v_src in zip(target_variables, source_variables)
            ]
        )
        session = session if session is not None else tf.get_default_session()
        return session.run(equal_op)

    def call(self, inputs):
        with tf.variable_scope(self.scope):
            x = self.W(inputs)
            return self.Q(x)


def compute_Q_loss(Q, Q2, R, D, A):
    selected_Q = tf.reduce_sum(
        Q * tf.one_hot(A, env.action_space.n), reduction_indices=[1]
    )
    next_Q = tf.math.reduce_max(Q2, axis=1)
    G = tf.stop_gradient(R + GAMMA * next_Q * (1 - D))
    q_loss = tf.reduce_sum(tf.square(selected_Q - G))
    return q_loss


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
R = tf.placeholder(dtype=tf.float32, shape=(None,), name="reward")  # reward
D = tf.placeholder(dtype=tf.float32, shape=(None,), name="done")  # done
A = tf.placeholder(dtype=tf.int32, shape=(None,), name="action")  # action_dim

# Convolution
conv_layer = ConvLayer()
Z = conv_layer(X)
Z2 = conv_layer(X2)

# Deep Q Network
q_layer = QLayer(output_dim=env.action_space.n, scope="main", trainable=True)
Q = q_layer(Z)
predict_op = tf.squeeze(tf.argmax(Q, axis=1))

target_q_layer = QLayer(output_dim=env.action_space.n, scope="target", trainable=False)
Q2 = target_q_layer(Z2)

q_loss = compute_Q_loss(Q, Q2, R, D, A)
train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(q_loss)

image_preprocessor = ImagePreprocessor(*env.observation_space.shape)

session = tf.Session()
session.run(tf.global_variables_initializer())
target_q_layer.copy_from(q_layer, session=session)
target_q_layer.is_equal(q_layer, session=session)
target_q_layer.update_from(q_layer, decay=0.9, session=session)

writer = tf.summary.FileWriter(LOGGING_DIR)
writer.add_graph(session.graph)

# Frame Stack


class FrameStack:
    def __init__(self, initial_frame, stack_size=STACK_SIZE):
        self.stack = deque(maxlen=stack_size)
        for _ in range(stack_size):
            self.stack.append(initial_frame)

    def append(self, frame):
        self.stack.append(frame)

    def get_state(self):
        stacked_state = np.stack(self.stack, axis=2)
        return stacked_state


#%% Replay Buffer


class SimpleReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        buffer_size = len(self.buffer)
        indices = np.random.choice(
            np.arange(buffer_size), size=batch_size, replace=False
        )
        size = len(indices)
        states = np.zeros([size, 210, 160, 4], dtype=np.float32)
        next_states = np.zeros([size, 210, 160, 4], dtype=np.float32)
        actions = np.zeros(size, dtype=np.float32)
        rewards = np.zeros(size, dtype=np.float32)
        dones = np.zeros(size, dtype=np.float32)

        for i, idx in enumerate(indices):
            state, action, reward, next_state, done = self.buffer[idx]
            states[i] = state
            next_states[i] = next_state
            actions[i] = action
            rewards[i] = reward
            dones[i] = done

        return dict(s=states, s2=next_states, a=actions, r=rewards, d=dones)

    def store(self, state, action, reward, next_state, done):
        self.add((state, action, reward, next_state, done))

    def number_of_samples(self):
        return len(self.buffer)


replay_buffer = SimpleReplayBuffer()
# %% Play


def sample_action(env, state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        feed_dict = {X: atleast_4d(state)}
        return session.run(predict_op, feed_dict)


def play_once(env, epsilon, render=False):
    observation = env.reset()
    done = False
    steps = 0
    total_return = 0
    frame = image_preprocessor.transform(observation, session)
    frame_stack = FrameStack(frame)
    while not done:
        state = frame_stack.get_state()

        action = sample_action(env, state, epsilon)
        observation, reward, done, _ = env.step(action)

        frame = image_preprocessor.transform(observation, session)
        frame_stack.append(frame)
        next_state = frame_stack.get_state()

        replay_buffer.store(state, action, reward, next_state, done)

        steps += 1
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
            X: batch["s"],
            X2: batch["s2"],
            A: np.squeeze(batch["a"]),
            R: batch["r"],
            D: batch["d"],
        }
        session.run(train_op, feed_dict)
        target_q_layer.update_from(q_layer, decay=DECAY, session=session)
        losses[n] = session.run(q_loss, feed_dict)
    return np.mean(losses)


# frame = image_preprocessor.transform(env.reset(), session=session)
# frame_stack = FrameStack(frame)
# feed_dict = {X: atleast_4d(frame_stack.get_state())}
# np.argmax(session.run(Q, feed_dict))
# session.run(tf.squeeze(tf.argmax(Q, axis=1)), feed_dict)

# %% main loop
returns = []

for n in range(ITERATIONS):
    epsilon = 1 / np.sqrt(n + 1)
    steps, total_return = play_once(env, epsilon, render=True)

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
for n in range(DEMO_NUMBER):
    play_once(test_env, 0.0, render=True)

# %%Close Environment
env.close()

#%% Report

plt.figure()
plt.plot(returns)
plt.title("Returns")
plt.savefig(os.path.join(monitor_dir, "returns.pdf"))
