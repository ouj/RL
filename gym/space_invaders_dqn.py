#!/usr/bin/env python3.7
import os
from datetime import datetime
import gym
import numpy as np
import tensorflow as tf
from rl.stacked_replay_buffer import StackedFrameReplayBuffer
from rl.helpers import set_random_seed
from rl.ann import create_hidden_layers, get_vars
import matplotlib.pyplot as plt

#%% Set random seeds
set_random_seed(0)

#%% Configuragtions
LEARNING_RATE = 0.001
BATCH_SIZE = 64
GAMMA = 0.99
DECAY = 0.99
MINIMAL_SAMPLES = 1000
ITERATIONS = 20000
DEMO_NUMBER = 10
STACK_SIZE = 4

#%% Setup
env = gym.make("SpaceInvaders-v4")

#%% Replay Buffer

replay_buffer = StackedFrameReplayBuffer(
    frame_width=env.observation_space.shape[0],
    frame_height=env.observation_space.shape[1],
    channels=env.observation_space.shape[2],
    stack_size=STACK_SIZE,
    action_dim=1,
)

def atleast_4d(x):
    return np.expand_dims(np.atleast_3d(x), axis=0) if x.ndim < 4 else x

#%% Reset

tf.reset_default_graph()

#%% Convolution Network

def create_conv_net(x):
    w = tf.layers.Conv2D(
        filters=32, kernel_size=8, activation=tf.nn.relu
    )(x)
    w = tf.layers.MaxPooling2D(pool_size=4, strides=4)(w)
    v = tf.layers.Conv2D(
        filters=64, kernel_size=4, activation=tf.nn.relu
    )(w)
    v = tf.layers.MaxPooling2D(pool_size=2, strides=2)(v)
    u = tf.layers.Conv2D(
        filters=64, kernel_size=3, activation=tf.nn.relu
    )(v)
    u = tf.layers.MaxPooling2D(pool_size=2, strides=2)(u)
    z = tf.layers.Flatten()(u)
    return z


with tf.variable_scope("conv") as scope:
    x = tf.placeholder(
        shape=(
            None,
            env.observation_space.shape[0],
            env.observation_space.shape[1],
            env.observation_space.shape[2] * STACK_SIZE
        ), dtype=tf.float32, name="x"
    )
    z = create_conv_net(x)


with tf.variable_scope(scope, reuse=True):
    x2 = tf.placeholder(
        shape=(
            None,
            env.observation_space.shape[0],
            env.observation_space.shape[1],
            env.observation_space.shape[2] * STACK_SIZE
        ), dtype=tf.float32, name="x2"
    )
    z2 = create_conv_net(x2)


#%% Q network

def create_dense_net(z, trainable=True):
    w = tf.keras.layers.Dense(
        units=500,
        activation=tf.nn.relu,
        trainable=trainable,
        name="w")(z)
    v = tf.keras.layers.Dense(
        units=500,
        activation=tf.nn.relu,
        trainable=trainable,
        name="v")(w)
    q = tf.keras.layers.Dense(
        units=env.action_space.n,
        trainable=trainable,
        name="q")(v)
    return q

w = tf.keras.layers.Dense(
        units=500,
        activation=tf.nn.relu,
        name="w")(z)

with tf.variable_scope("main"):
    q = create_dense_net(z, trainable=True)

with tf.variable_scope("target"):
    q2 = create_dense_net(z2, trainable=False)

r = tf.placeholder(dtype=tf.float32, shape=(None,)) # reward
d = tf.placeholder(dtype=tf.float32, shape=(None,)) # done
a = tf.placeholder(dtype=tf.int32, shape=(None,)) # action

selected_q = tf.reduce_sum(
    q * tf.one_hot(a, env.action_space.n),
    reduction_indices=[1]
)

next_q = tf.math.reduce_max(q2, axis=1)
g = tf.stop_gradient(
    r + GAMMA * next_q * (1 - d)
)
q_loss = tf.reduce_sum(tf.square(selected_q - g))

train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(
    q_loss
)

# Copy main network params to target networks
target_init = tf.group([
    tf.assign(v_targ, v_main)
    for v_main, v_targ in zip(get_vars('main'), get_vars('target'))
])

# Use soft updates to update the target networks
target_update = tf.group([
    tf.assign(v_targ, DECAY * v_targ + (1 - DECAY) * v_main)
    for v_main, v_targ in zip(get_vars('main'), get_vars('target'))
])

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
session.run(target_init)

# %% Play

def update_state(state, observation):
    return np.concatenate((state[:,:,STACK_SIZE-1:], observation), axis=2)

def sample_action(env, state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        feed_dict={
            x: atleast_4d(state)
        }
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
def train():
    for _ in range(steps):
        batch = replay_buffer.sample_batch(BATCH_SIZE)

        feed_dict = {
            x: batch['s'],
            x2: batch['s2'],
            a: np.squeeze(batch['a']),
            r: batch['r'],
            d: batch['d']
        }
        session.run(train_op, feed_dict)
        session.run(target_update)
        return session.run(q_loss, feed_dict)

# %% main loop
losses = []
returns = []

for n in range(ITERATIONS):
    epsilon = 1 / np.sqrt(n+1)
    steps, total_return = play_once(env, epsilon, render=False)

    returns.append(total_return)
    if MINIMAL_SAMPLES < replay_buffer.number_of_samples():
        loss = train()
        losses.append(loss)

    if n != 0 and n % 10 == 0:
        print(
            "Episode:", n,
            "Average Returns:", np.mean(returns[n-10:]),
            "epsilon:", epsilon
        )

#%% Demo

filename = os.path.basename(__file__).split('.')[0]
monitor_dir = './' + filename + '_' + str(datetime.now())
env = gym.wrappers.Monitor(env, monitor_dir)
for n in range(DEMO_NUMBER):
    play_once(env, 0.0, render=True)

# %%Close Environment
env.close()

#%% Report
plt.figure()
plt.plot(losses)
plt.title("Q Losses")
plt.savefig(os.path.join(monitor_dir, 'q_loss.pdf'))

plt.figure()
plt.plot(returns)
plt.title("Returns")
plt.savefig(os.path.join(monitor_dir, 'returns.pdf'))
