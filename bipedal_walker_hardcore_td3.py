#!/usr/bin/env python3

import os
from datetime import datetime

import gym
import numpy as np
import tensorflow as tf
from common.helpers import set_random_seed, get_saver
from common.mlp import MLPNetwork
from common.schedules import LinearSchedule

set_random_seed(0)

# Path and folders
FILENAME = "bipedal_walker_hardcore_td3"
TS = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
MONITOR_DIR = os.path.join("output", FILENAME, "video", TS)
LOGGING_DIR = os.path.join("output", FILENAME, "log", "run2")
CHECKPOINT_DIR = os.path.join("output", FILENAME, "checkpoints")

# Hyperparameters
MU_LEARNING_RATE = 1e-3
Q_LEARNING_RATE = 1e-3
GAMMA = 0.99
DECAY = 0.995
ACTION_NOISE = 0.1
POLICY_NOISE = 0.2
POLICY_NOISE_CLIP = 0.5
POLICY_FREQ = 2
MINIMAL_SAMPLES = 10000
MAXIMAL_SAMPLES = 1000000
ITERATIONS = 100000
BATCH_SIZE = 32

MAX_EPISODE_LENGTH = 1600

SAVE_CHECKPOINT_EVERY = 100
DEMO_EVERY = 25

# Environment
env = gym.make("BipedalWalkerHardcore-v2")


def create_mu_net(
    name, output_dim, action_max, activation=tf.nn.relu, trainable=True
):
    return MLPNetwork([
        tf.layers.Dense(
            units=512,
            activation=activation,
            trainable=trainable,
            name="H1",
            kernel_initializer=tf.initializers.glorot_normal(),
        ),
        tf.layers.Dense(
            units=256,
            activation=activation,
            trainable=trainable,
            name="H2",
            kernel_initializer=tf.initializers.glorot_normal(),
        ),
        tf.layers.Dense(
            units=output_dim,
            activation=tf.nn.tanh,
            trainable=trainable,
            name="O",
            kernel_initializer=tf.initializers.glorot_normal(),
        ),
        tf.keras.layers.Lambda(
            lambda x: action_max * x,
            name="action_max",
        ),
    ], name=name)


def create_q_net(name, activation=tf.nn.relu, trainable=True):
    return MLPNetwork([
        tf.layers.Dense(
            units=512,
            activation=activation,
            trainable=trainable,
            name="H1",
            kernel_initializer=tf.initializers.glorot_normal,
        ),
        tf.layers.Dense(
            units=256,
            activation=activation,
            trainable=trainable,
            name="H2",
            kernel_initializer=tf.initializers.glorot_normal(),
        ),
        tf.layers.Dense(
            units=1,
            trainable=trainable,
            name="O",
            kernel_initializer=tf.initializers.glorot_normal,
        ),
        tf.keras.layers.Lambda(
            lambda x: tf.squeeze(x),
            name="squeeze",
        ),
    ], name=name)


def add_clipped_noise(action, action_max):
    noise = tf.random_normal(
        shape=tf.shape(action),
        mean=0.0,
        stddev=POLICY_NOISE,
        dtype=tf.float32
    )
    noise = tf.clip_by_value(noise, -POLICY_NOISE_CLIP, POLICY_NOISE_CLIP)
    action = tf.clip_by_value(tf.math.add(
        action, noise), -action_max, action_max)
    return action


tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)

X = tf.placeholder(
    shape=(None, env.observation_space.shape[0]), dtype=tf.float32, name="X"
)
X2 = tf.placeholder(
    shape=(None, env.observation_space.shape[0]), dtype=tf.float32, name="X2"
)
A = tf.placeholder(
    dtype=tf.float32, shape=(None, *env.action_space.shape), name="ACTION"
)  # action
R = tf.placeholder(dtype=tf.float32, shape=(None,), name="REWARD")  # reward
D = tf.placeholder(dtype=tf.float32, shape=(None,), name="DONE")  # done

action_dim = env.action_space.shape[0]
action_max = env.action_space.high[0]
action_min = env.action_space.low[0]

mu_net = create_mu_net(
    name="Mu/main",
    output_dim=action_dim,
    action_max=action_max,
    trainable=True
)
targ_mu_net = create_mu_net(
    name="Mu/target",
    output_dim=action_dim,
    action_max=action_max,
    trainable=False
)

q1_net = create_q_net(name="Q1/main", trainable=True)
q2_net = create_q_net(name="Q2/main", trainable=True)
targ_q1_net = create_q_net(name="Q1/target", trainable=False)
targ_q2_net = create_q_net(name="Q2/target", trainable=False)

mu = mu_net(X)
X_A = tf.concat([X, A], axis=-1, name="XA")

Q1 = q1_net(X_A)
Q2 = q2_net(X_A)

A2 = add_clipped_noise(targ_mu_net(X2), action_max)
X2_A2 = tf.concat([X2, A2], axis=-1, name="X2_A2")

next_Q1 = targ_q1_net(X2_A2)
next_Q2 = targ_q2_net(X2_A2)
target_next_Q = tf.math.minimum(next_Q1, next_Q2, name="target_next_Q")

with tf.name_scope("GlobalStep"):
    global_step = tf.train.get_or_create_global_step()
    global_step_op = tf.assign_add(global_step, 1, name="increment")

with tf.name_scope("Prediction"):
    predict_op = mu

with tf.name_scope("Training"):
    with tf.name_scope("Q"):
        G = tf.stop_gradient(R + GAMMA * (1 - D) * target_next_Q)
        q1_loss = tf.reduce_mean(tf.math.square(Q1 - G))
        q2_loss = tf.reduce_mean(tf.math.square(Q2 - G))

    with tf.name_scope("Mu"):
        X_mu = tf.concat([X, mu], axis=-1)
        Q1_mu = q1_net(X_mu)  # For policy gradient
        mu_loss = -tf.reduce_mean(Q1_mu)

    with tf.name_scope("TrainOp"):
        q1_train_op = tf.train.AdamOptimizer(learning_rate=Q_LEARNING_RATE).minimize(
            q1_loss, var_list=q1_net.collect_variables()
        )
        q2_train_op = tf.train.AdamOptimizer(learning_rate=Q_LEARNING_RATE).minimize(
            q2_loss, var_list=q2_net.collect_variables()
        )
        mu_train_op = tf.train.AdamOptimizer(learning_rate=MU_LEARNING_RATE).minimize(
            mu_loss, var_list=mu_net.collect_variables()
        )

    with tf.name_scope("Summary"):
        tf.summary.histogram("Actor 1", Q1)
        tf.summary.histogram("Actor 2", Q2)
        tf.summary.histogram("Critic", G)
        tf.summary.histogram("Advantage 1", Q1 - G)
        tf.summary.histogram("Advantage 2", Q2 - G)
        tf.summary.scalar("Q1_Loss", q1_loss)
        tf.summary.scalar("Q2_Loss", q2_loss)
        tf.summary.scalar("Mu_Loss", mu_loss)

with tf.name_scope("CopyOp"):
    copy_op = tf.group([
        targ_mu_net.copy_from(mu_net),
        targ_q1_net.copy_from(q1_net),
        targ_q2_net.copy_from(q2_net)
    ])

with tf.name_scope("UpdateOp"):
    update_op = tf.group([
        targ_mu_net.update_from(mu_net, decay=DECAY),
        targ_q1_net.update_from(q1_net, decay=DECAY),
        targ_q2_net.update_from(q2_net, decay=DECAY)
    ])

# Initialize Session and Run copy ops
session = tf.Session()
init_op = tf.global_variables_initializer()
session.run(init_op)
session.run(copy_op)

# Setup Tensorboard
for net in [mu_net, targ_mu_net, q1_net, q2_net, targ_q1_net, targ_q2_net]:
    net.setup_tensorboard()
summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter(LOGGING_DIR)
writer.add_graph(session.graph)

# Saver
saver = get_saver(session, CHECKPOINT_DIR)


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
        session.run([q1_train_op, q2_train_op, global_step_op], feed_dict)

        if n != 0 and n % POLICY_FREQ == 0:
            session.run(mu_train_op, feed_dict)
            session.run(update_op)

    return session.run(summary_op, feed_dict)


def demo():
    demo_env = gym.wrappers.Monitor(
        env, MONITOR_DIR, resume=True, mode="evaluation", write_upon_reset=True
    )
    steps, total_return = play_once(demo_env, random_action=False, render=True)
    print("Demo for %d steps, Return %d" % (steps, total_return))
    summary = tf.Summary()
    summary.value.add(tag="Demo/Return", simple_value=total_return)
    summary.value.add(tag="Demo/Steps", simple_value=steps)
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
