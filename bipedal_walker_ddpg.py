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
FILENAME = "bipedal_walker_ddpg"
TS = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
MONITOR_DIR = os.path.join("output", FILENAME, "video", TS)
LOGGING_DIR = os.path.join("output", FILENAME, "log", "run5")
CHECKPOINT_DIR = os.path.join("output", FILENAME, "checkpoints")

# Hyperparameters
MU_LEARNING_RATE = 1e-3
Q_LEARNING_RATE = 1e-3
GAMMA = 0.99
DECAY = 0.995
ACTION_NOISE = 0.1
MINIMAL_SAMPLES = 10000
MAXIMAL_SAMPLES = 1000000
ITERATIONS = 100000
BATCH_SIZE = 32

MAX_EPISODE_LENGTH = 1600

SAVE_CHECKPOINT_EVERY = 100
DEMO_EVERY = 100

# Environment
env = gym.make("BipedalWalker-v2")


def create_mu_network(
    name,
    output_dim,
    action_max,
    activation=tf.nn.relu,
    output_activation=tf.nn.tanh,
    trainable=True,
):
    return MLPNetwork([
        tf.layers.Dense(
            units=512,
            activation=activation,
            trainable=trainable,
            name="W",
            kernel_initializer=tf.initializers.glorot_normal,
        ),
        tf.layers.Dense(
            units=output_dim,
            activation=output_activation,
            trainable=trainable,
            name="MU",
            kernel_initializer=tf.initializers.glorot_normal,
        ),
        tf.keras.layers.Lambda(
            lambda x: action_max * x,
            name="action_max",
        ),
    ], name=name)


def create_q_network(name, activation=tf.nn.relu, trainable=True):
    return MLPNetwork([
        tf.layers.Dense(
            units=512,
            activation=activation,
            trainable=trainable,
            name="W",
            kernel_initializer=tf.initializers.glorot_normal,
        ),
        tf.layers.Dense(
            units=1,
            trainable=trainable,
            name="Q",
            kernel_initializer=tf.initializers.glorot_normal,
        ),
        tf.keras.layers.Lambda(
            lambda x: tf.squeeze(x),
            name="squeeze",
        ),
    ], name=name)


tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)

X = tf.placeholder(
    shape=(None, env.observation_space.shape[0]), dtype=tf.float32, name="x"
)
X2 = tf.placeholder(
    shape=(None, env.observation_space.shape[0]), dtype=tf.float32, name="x2"
)
A = tf.placeholder(
    dtype=tf.float32, shape=(None, *env.action_space.shape), name="action"
)  # action
R = tf.placeholder(dtype=tf.float32, shape=(None,), name="reward")  # reward
D = tf.placeholder(dtype=tf.float32, shape=(None,), name="done")  # done

action_dim = env.action_space.shape[0]
action_max = env.action_space.high[0]

mu_network = create_mu_network(
    name="main/mu", output_dim=action_dim, action_max=action_max, trainable=True)
q_network = create_q_network(name="main/Q", trainable=True)

target_mu_network = create_mu_network(
    name="target/mu", output_dim=action_dim, action_max=action_max, trainable=False)
target_q_network = create_q_network(name="target/Q", trainable=False)

mu = mu_network(X)
Q = q_network(tf.concat([X, A], axis=-1))
Q_mu = q_network(tf.concat([X, mu], axis=-1))

next_mu = target_mu_network(X2)
next_Q = target_q_network(tf.concat([X2, next_mu], axis=-1))

with tf.name_scope("global_step"):
    global_step = tf.train.get_or_create_global_step()
    global_step_op = tf.assign_add(global_step, 1, name="increment")

with tf.name_scope("predict_op"):
    predict_op = mu

with tf.name_scope("q_train_op"):
    G = tf.stop_gradient(R + GAMMA * (1 - D) * next_Q)
    # https://openai.com/blog/openai-baselines-dqn/ suggest huber_loss
    # q_loss = tf.reduce_mean(tf.losses.huber_loss(Q, G))
    q_loss = tf.reduce_mean(tf.math.square(Q - G))
    q_train_op = tf.train.AdamOptimizer(learning_rate=Q_LEARNING_RATE).minimize(
        q_loss, var_list=q_network.collect_variables()
    )
    tf.summary.histogram("G", G)
    tf.summary.scalar("G_mean", tf.reduce_mean(G))
    tf.summary.scalar("Q_loss", q_loss)

with tf.name_scope("mu_train_op"):
    mu_loss = -tf.reduce_mean(Q_mu)
    mu_train_op = tf.train.AdamOptimizer(learning_rate=MU_LEARNING_RATE).minimize(
        mu_loss, var_list=mu_network.collect_variables()
    )
    tf.summary.scalar("Mu_loss", mu_loss)

with tf.name_scope("copy_op"):
    copy_op = tf.group(
        [target_mu_network.copy_from(
            mu_network), target_q_network.copy_from(q_network)]
    )

with tf.name_scope("update_op"):
    update_op = tf.group(
        [
            target_mu_network.update_from(mu_network, decay=DECAY),
            target_q_network.update_from(q_network, decay=DECAY),
        ]
    )


class ReplayBuffer:
    def __init__(self, observation_dim, action_dim, max_size=1000000):
        self.index = 0
        self.size = 0
        self.max_size = max_size
        self.states = np.zeros([max_size, observation_dim], dtype=np.float32)
        self.next_states = np.zeros(
            [max_size, observation_dim], dtype=np.float32)
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
    observation_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    max_size=MAXIMAL_SAMPLES
)

# Initialize Session and Run copy ops
session = tf.Session()
init_op = tf.global_variables_initializer()
session.run(init_op)
session.run(copy_op)

# Setup Summary
mu_network.setup_tensorboard()
target_mu_network.setup_tensorboard()
q_network.setup_tensorboard()
target_q_network.setup_tensorboard()
summary_op = tf.summary.merge_all()

writer = tf.summary.FileWriter(LOGGING_DIR)
writer.add_graph(session.graph)

# Saver
saver = tf.train.Saver()
last_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
if last_checkpoint is not None:
    saver.restore(session, last_checkpoint)
    print("Restored last checkpoint", last_checkpoint)

# Play


def get_action(observation):
    feed_dict = {X: np.atleast_2d(observation)}
    action = session.run(predict_op, feed_dict)[0]
    action += ACTION_NOISE * np.random.randn(action_dim)
    return np.clip(action, -action_max, action_max)


def play_once(env, random_action, max_steps=MAX_EPISODE_LENGTH, render=False):
    observation = env.reset()
    done = False
    steps = 0
    total_return = 0
    while not done:
        if random_action:
            action = env.action_space.sample()
        else:
            action = get_action(observation)
        next_observation, reward, done, _ = env.step(action)

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d_store = False if steps == max_steps else done
        replay_buffer.store(observation, action, reward,
                            next_observation, d_store)

        observation = next_observation
        steps += 1
        total_return += reward
        if render:
            env.render()
    return steps, total_return


# Train
def train(steps):
    for n in range(steps):
        batch = replay_buffer.sample_batch(batch_size=BATCH_SIZE)
        feed_dict = {
            X: batch["s"],
            X2: batch["s2"],
            A: batch["a"],
            R: batch["r"],
            D: batch["d"],
        }
        session.run(q_train_op, feed_dict)
        session.run(mu_train_op, feed_dict)
        session.run([update_op, global_step_op])
    return session.run(summary_op, feed_dict)


def demo():
    demo_env = gym.wrappers.Monitor(
        env, MONITOR_DIR, resume=True, mode="evaluation", write_upon_reset=True
    )
    steps, total_return = play_once(demo_env, random_action=False, render=True)
    print("Demo for %d steps, Return %d" % (steps, total_return))
    summary = tf.Summary()
    summary.value.add(tag="demo/return", simple_value=total_return)
    summary.value.add(tag="demo/steps", simple_value=steps)
    demo_env.close()
    return summary


# Populate replay buffer
print("Populating replay buffer")
while MINIMAL_SAMPLES > replay_buffer.number_of_samples():
    steps, total_return = play_once(env, random_action=True, render=False)
    print("Played %d < %d steps" %
          (replay_buffer.number_of_samples(), MINIMAL_SAMPLES))

# Main loop
print("Start Main Loop...")
for n in range(ITERATIONS):
    gstep = tf.train.global_step(session, global_step)
    steps, total_return = play_once(env, random_action=False)
    t0 = datetime.now()
    train_summary = train(steps)
    delta = datetime.now() - t0
    print(
        "Episode:",
        n,
        "Return:",
        total_return,
        "Step:",
        steps,
        "Duration:",
        delta.total_seconds(),
        "Global Steps:",
        gstep,
    )

    summary = tf.Summary()
    summary.value.add(tag="misc/return", simple_value=total_return)
    summary.value.add(tag="misc/steps", simple_value=steps)
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

# Close Environment
env.close()
