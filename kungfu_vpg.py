#!/usr/bin/env python3.7

# TODO:
# 1. Figure out the initializers for conv net so that we might not need batch norm.
# 2. Use dataset to feed the data so that we don't have OOM
# 3. Exam the output V and Q to make sure they are have sensible values

import gym
import retro
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from common.helpers import set_random_seed, atleast_4d
from common.mlp import MLPNetwork
from matplotlib import pyplot as plt

# Set random seeds
set_random_seed(0)

# Path and folders
FILENAME = "kungfu_vpg"
TS = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
MONITOR_DIR = os.path.join("output", FILENAME, "video", TS)
LOGGING_DIR = os.path.join("output", FILENAME, "log", "run1")
CHECKPOINT_DIR = os.path.join("output", FILENAME, "checkpoints")

# Hyperparameters
P_LEARNING_RATE = 1e-3
V_LEARNING_RATE = 1e-3
GAMMA = 0.9999
ITERATIONS = 10000

SAVE_CHECKPOINT_EVERY = 100
DEMO_EVERY = 10

env = retro.make(game='KungFu-Nes',
                 use_restricted_actions=retro.Actions.DISCRETE)

tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)

session = tf.Session()

# Image preprocessing


class ImagePreprocessor:
    def __init__(self, session):
        self.session = session
        with tf.variable_scope("image_preprocessor"):
            self.input = tf.placeholder(shape=[224, 240, 3], dtype=tf.uint8)
            t = tf.image.rgb_to_grayscale(self.input)
            t = tf.image.convert_image_dtype(t, dtype=tf.float32)
            t = tf.image.crop_to_bounding_box(t, 100, 0, 80, 240)
            self.output = t

    def set_session(self, session):
        self.session = session

    def transform(self, frame):
        return self.session.run(self.output, feed_dict={self.input: frame})


image_processor = ImagePreprocessor(session)

X = tf.placeholder(tf.float32, shape=(None, 80, 240, 1), name='X')
G = tf.placeholder(tf.float32, shape=(None,), name='G')
A = tf.placeholder(tf.int32, shape=(None,), name='actions')


def create_conv_net(activation=tf.nn.relu):
    return MLPNetwork([
        tf.layers.Conv2D(
            filters=32,
            kernel_size=8,
            strides=4,
            padding="valid",
            activation=activation,
            kernel_initializer=tf.initializers.glorot_uniform,
            name="conv1",
            use_bias=False,
        ),
        tf.layers.BatchNormalization(
            epsilon=1e-5,
            name="batch_norm1",
        ),
        tf.layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=2,
            padding="valid",
            activation=activation,
            kernel_initializer=tf.initializers.glorot_uniform,
            name="conv2",
            use_bias=False
        ),
        tf.layers.BatchNormalization(
            epsilon=1e-5,
            name="batch_norm2",
        ),
        tf.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding="valid",
            activation=activation,
            kernel_initializer=tf.initializers.glorot_uniform,
            name="conv3",
            use_bias=True,
        )
    ], name="ConvNet")


def create_q_net(output_dim, activation=tf.nn.relu, trainable=True):
    return MLPNetwork([
        tf.layers.Flatten(name="flatten"),
        tf.layers.Dense(
            units=512, activation=activation, trainable=trainable, name="W",
            kernel_initializer=tf.initializers.glorot_uniform,
        ),
        tf.layers.Dense(
            units=output_dim, trainable=trainable, name="Q",
            activation=tf.nn.softmax,
            kernel_initializer=tf.initializers.glorot_uniform
        )
    ], name="QNet")


def create_v_net(output_dim, activation=tf.nn.relu, trainable=True):
    return MLPNetwork([
        tf.layers.Flatten(name="flatten"),
        tf.layers.Dense(
            units=512, activation=activation, trainable=trainable, name="W",
            kernel_initializer=tf.initializers.glorot_uniform,
        ),
        tf.layers.Dense(
            units=output_dim, trainable=trainable, name="Q",
            kernel_initializer=tf.initializers.glorot_uniform
        ),
        tf.keras.layers.Lambda(
            lambda x: tf.math.reduce_max(x, axis=1),
            name="reduce_max"),
    ], name="VNet")


conv_net = create_conv_net()
q_net = create_q_net(env.action_space.n)
v_net = create_v_net(env.action_space.n)

Z = conv_net(X)
Q = q_net(Z)
V = v_net(Z)

with tf.name_scope("predict_op"):
    predict_op = tf.squeeze(Q)

with tf.name_scope("q_train_op"):
    advantages = tf.stop_gradient(G - V)
    tf.summary.histogram("advantages", advantages)
    selected_prob = tf.log(tf.reduce_sum(
        Q * tf.one_hot(A, env.action_space.n), reduction_indices=[1]
    ))
    q_loss = -tf.reduce_sum(advantages * selected_prob)
    tf.summary.scalar("q_loss", q_loss)
    q_train_op = tf.train.AdamOptimizer(
        learning_rate=P_LEARNING_RATE).minimize(q_loss)

with tf.name_scope("v_train_op"):
    tf.summary.histogram("V", V)
    v_loss = tf.reduce_sum(tf.square(G - V))
    tf.summary.scalar("v_loss", v_loss)
    v_train_op = tf.train.AdamOptimizer(V_LEARNING_RATE).minimize(v_loss)

with tf.name_scope("global_step"):
    global_step = tf.train.get_or_create_global_step()
    global_step_op = tf.assign_add(global_step, 1, name="increment")

# Initialize variables
session.run(tf.global_variables_initializer())

# Setup Summary
conv_net.setup_tensorboard()
q_net.setup_tensorboard()
v_net.setup_tensorboard()
summary_op = tf.summary.merge_all()

writer = tf.summary.FileWriter(LOGGING_DIR)
writer.add_graph(session.graph)

# Saver
saver = tf.train.Saver()
last_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
if last_checkpoint is not None:
    saver.restore(session, last_checkpoint)
    print("Restored last checkpoint", last_checkpoint)


def compute_returns(rewards):
    # compute returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.append(G)
    returns.reverse()
    return returns


def sample_action(state):
    p = session.run(predict_op, feed_dict={
        X: atleast_4d(state)
    })
    return np.random.choice(len(p), p=p)


def play(env, render=False):
    observation = env.reset()
    state = image_processor.transform(observation)
    done = False
    reward = 0

    states = []
    actions = []
    rewards = []
    while not done:
        action = sample_action(state)
        observation, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        # update state
        state = image_processor.transform(observation)
        if render:
            env.render()
    total_rewards, total_steps = np.sum(rewards), len(rewards)
    returns = compute_returns(rewards)
    return total_steps, total_rewards, np.asarray(
        states), np.asarray(actions), np.asarray(returns)


# Train
def train(states, actions, returns, epoches=1):
    feed_dict = {
        A: actions,
        X: states,
        G: returns
    }
    for e in range(epoches):
        session.run(
            [q_train_op, v_train_op, global_step_op], feed_dict=feed_dict
        )
    return session.run(summary_op, feed_dict)


def demo():
    # demo_env = gym.wrappers.Monitor(
    #     env, MONITOR_DIR, resume=True, mode="evaluation", write_upon_reset=True
    # )
    demo_env = env
    steps, total_return, _, _, _ = play(demo_env, render=True)
    print("Demo for %d steps, Return %d" % (steps, total_return))
    summary = tf.Summary()
    summary.value.add(tag="demo/return", simple_value=total_return)
    summary.value.add(tag="demo/steps", simple_value=steps)
    # demo_env.close()
    return summary


print("Start Main Loop...")
for n in range(ITERATIONS):
    gstep = tf.train.global_step(session, global_step)
    total_steps, total_rewards, states, actions, returns = play(
        env, render=False)
    t0 = datetime.now()
    train_summary = train(states, actions, returns)
    delta = datetime.now() - t0
    print(
        "Episode:",
        n,
        "Return:",
        total_rewards,
        "Step:",
        total_steps,
        "Duration:",
        delta.total_seconds(),
        "Global Steps:",
        gstep
    )

    summary = tf.Summary()
    summary.value.add(tag="misc/return", simple_value=total_rewards)
    summary.value.add(tag="misc/steps", simple_value=total_steps)
    summary.value.add(tag="misc/duration", simple_value=delta.total_seconds())
    writer.add_summary(train_summary, global_step=gstep)
    writer.add_summary(summary, global_step=gstep)

    if n != 0 and n % SAVE_CHECKPOINT_EVERY == 0:
        path = saver.save(
            session, os.path.join(CHECKPOINT_DIR, "model"), global_step=gstep
        )
        print("Saved checkpoint to", path)

    if n % DEMO_EVERY == 0:
        summary = demo()
        writer.add_summary(summary, global_step=gstep)


env.close()
