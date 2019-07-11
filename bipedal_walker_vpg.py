#!/usr/bin/env python3

import os
from datetime import datetime

import gym
import numpy as np
import tensorflow as tf
from common.helpers import set_random_seed
from common.mlp import MLPNetwork
from common.schedules import LinearSchedule


set_random_seed(0)

# Path and folders
FILENAME = "bipedal_walker_vpg"
TS = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
MONITOR_DIR = os.path.join("output", FILENAME, "video", TS)
LOGGING_DIR = os.path.join("output", FILENAME, "log", "run5")
CHECKPOINT_DIR = os.path.join("output", FILENAME, "checkpoints")

# Hyperparameters
P_LEARNING_RATE = 1e-5
V_LEARNING_RATE = 1e-5
GAMMA = 0.9999
ITERATIONS = 10000
SAVE_CHECKPOINT_EVERY = 100
DEMO_EVERY = 10
MAX_EPISODE_LENGTH = 1600

SAVE_CHECKPOINT_EVERY = 100
DEMO_EVERY = 100

# Environment
env = gym.make("BipedalWalker-v2")

tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)

session = tf.Session()

X = tf.placeholder(
    shape=(None, env.observation_space.shape[0]), dtype=tf.float32, name="x"
)
A = tf.placeholder(
    dtype=tf.float32, shape=(None, *env.action_space.shape), name="action"
)  # action
G = tf.placeholder(tf.float32, shape=(None,), name='G')

action_dim = env.action_space.shape[0]
action_max = env.action_space.high[0]

def create_policy_nets(output_dim, activation=tf.nn.relu):
    with tf.name_scope("mean"):
        mean_net = MLPNetwork([
            tf.layers.Dense(
                units=256, activation=activation, name="W",
                kernel_initializer=tf.initializers.glorot_uniform(),
                use_bias=False
            ),
            tf.layers.Dense(
                units=output_dim, name="Q",
                activation=tf.nn.softmax,
                kernel_initializer=tf.initializers.glorot_uniform(),
                use_bias=False
            ),
            tf.keras.layers.Lambda(
                lambda x: tf.squeeze(x),
                name="squeeze",
            ),
        ], name="policy_mean")

    with tf.name_scope("stddev"):
        stddev_net = MLPNetwork([
            tf.layers.Dense(
                units=256, activation=activation, name="W",
                kernel_initializer=tf.initializers.glorot_uniform(),
                use_bias=False
            ),
            tf.layers.Dense(
                units=output_dim, name="Q",
                activation=tf.nn.softplus,
                kernel_initializer=tf.initializers.glorot_uniform(),
                use_bias=False
            ),
            tf.keras.layers.Lambda(
                lambda x: tf.squeeze(x) + 1e-5,
                name="squeeze_smoothing",
            ),
        ], name="policy_stddev")

    return mean_net, stddev_net


def create_v_net(output_dim, activation=tf.nn.relu):
    return MLPNetwork([
        tf.layers.Dense(
            units=256, activation=activation, name="W",
            kernel_initializer=tf.initializers.glorot_uniform(),
        ),
        tf.layers.Dense(
            units=1, name="Q",
            kernel_initializer=tf.initializers.glorot_uniform(),
        ),
    ], name="VNet")

policy_mean_net, policy_stddev_net = create_policy_nets(action_dim, )
v_net = create_v_net(action_dim, )

mean = policy_mean_net(X)
stddev = policy_stddev_net(X)
V = v_net(X)

with tf.name_scope("global_step"):
    global_step = tf.train.get_or_create_global_step()
    global_step_op = tf.assign_add(global_step, 1, name="increment")

with tf.name_scope("predict_op"):
    norm = tf.distributions.Normal(mean, stddev)
    predict_op = tf.clip_by_value(norm.sample(), -1, 1)

with tf.name_scope("train_op"):
    with tf.name_scope("Policy"):
        advantages = tf.stop_gradient(G - V)
        log_probs = norm.log_prob(A)
        cost = -tf.reduce_sum(A * log_probs + 0.1 * norm.entropy())
        train_op = tf.train.AdamOptimizer(P_LEARNING_RATE).minimize(cost)
        tf.summary.scalar("cost", cost)

    with tf.name_scope("Value"):
        v_loss = tf.reduce_sum(tf.square(G - V))
        v_train_op = tf.train.AdamOptimizer(V_LEARNING_RATE).minimize(v_loss)
        tf.summary.histogram("value", V)
        tf.summary.scalar("max_value", tf.math.reduce_max(V))

# Setup Summary
policy_mean_net.setup_tensorboard()
policy_stddev_net.setup_tensorboard()
v_net.setup_tensorboard()
summary_op = tf.summary.merge_all()

# Initialize Session and Run copy ops
session.run(tf.global_variables_initializer())

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


def sample_action(observation):
    return session.run(predict_op, feed_dict={
        X: np.atleast_2d(observation)
    })

def play(env, render=False):
    observation = env.reset()
    done = False
    reward = 0

    states = []
    actions = []
    rewards = []
    while not done:
        action = sample_action(observation)
        prev_observation = observation
        observation, reward, done, _ = env.step(action)
        states.append(prev_observation)
        actions.append(action)
        rewards.append(reward)
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
            [train_op, v_train_op, global_step_op], feed_dict=feed_dict
        )
    return session.run(summary_op, feed_dict)


def demo():
    steps, total_return, _, _, _ = play(env, render=True)
    print("Demo for %d steps, Return %d" % (steps, total_return))
    summary = tf.Summary()
    summary.value.add(tag="demo/return", simple_value=total_return)
    summary.value.add(tag="demo/steps", simple_value=steps)
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
