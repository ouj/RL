#!/usr/bin/env python3.7
import os
from datetime import datetime
import gym
import numpy as np
import tensorflow as tf
from rl.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

#%% Set random seeds
def set_random_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
set_random_seed(0)

#%% Helper functions
def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]


#%% Configuragtions
LEARNING_RATE = 0.001
BATCH_SIZE = 64
GAMMA = 0.99
DECAY = 0.99
MINIMAL_SAMPLES = 1000
ITERATIONS = 20000
DEMO_NUMBER = 10

#%% Setup
env = gym.make("CartPole-v1")


#%% Q network
# Reset tf
tf.reset_default_graph()

with tf.variable_scope("main"):
    x = tf.placeholder(
        shape=(None, env.observation_space.shape[0]),
        dtype=tf.float32,
        name="x"
    )
    w = tf.keras.layers.Dense(units=20, activation=tf.nn.relu, name="w")(x)
    v = tf.keras.layers.Dense(units=20, activation=tf.nn.relu, name="v")(w)
    q = tf.keras.layers.Dense(units=env.action_space.n, name="q")(v)

with tf.variable_scope("target"):
    x2 = tf.placeholder(
        shape=(None, env.observation_space.shape[0]),
        dtype=tf.float32,
        name="x2"
    )
    w2 = tf.keras.layers.Dense(
        units=20, activation=tf.nn.relu, name="w", trainable=False
    )(x2)
    v2 = tf.keras.layers.Dense(
        units=20, activation=tf.nn.relu, name="v", trainable=False
    )(w2)
    q_target = tf.keras.layers.Dense(
        units=env.action_space.n, name="q", trainable=False
    )(v2)


r = tf.placeholder(dtype=tf.float32, shape=(None,)) # reward
d = tf.placeholder(dtype=tf.float32, shape=(None,)) # done
a = tf.placeholder(dtype=tf.int32, shape=(None,)) # action

selected_q = tf.reduce_sum(
    q * tf.one_hot(a, env.action_space.n),
    reduction_indices=[1]
)

next_q = tf.math.reduce_max(q_target, axis=1)
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

#%% Replay Buffer

replay_buffer = ReplayBuffer(
    observation_shape=env.observation_space.shape,
    action_shape=(1, )
)

# %% Play
def sample_action(env, observation, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        q_s_a = session.run(q, feed_dict={
            x: np.atleast_2d(observation)
        })[0]
        return np.argmax(q_s_a)

def play_once(env, epsilon, render=False):
    observation = env.reset()
    done = False
    steps = 0
    total_return = 0
    while not done:
        action = sample_action(env, observation, epsilon)
        next_observation, reward, done, _ = env.step(action)
        replay_buffer.store(observation, action, reward, next_observation, done)
        observation = next_observation
        steps += 1
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
    epsilon = 1.0 / np.sqrt(n+1)
    steps, total_return = play_once(env, epsilon)

    returns.append(total_return)
    if MINIMAL_SAMPLES < replay_buffer.number_of_samples():
        loss = train()
        losses.append(loss)

    if n != 0 and n % 10 == 0:
        print(
            "Episode:", n,
            "Returns:", total_return,
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
