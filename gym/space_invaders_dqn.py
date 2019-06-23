#!/usr/bin/env python3.7
import os
import sys
from datetime import datetime
import gym
import numpy as np
import tensorflow as tf
from rl.helpers import atleast_4d, set_random_seed
from rl.stacked_frame_replay_buffer import StackedFrameReplayBuffer
from wrappers.atari_wrappers import EpisodicLifeEnv


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set random seeds
set_random_seed(0)

# Path and folders
FILENAME = "space_invadors_dpn"
TS = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
MONITOR_DIR = os.path.join("output", FILENAME, "video", TS)
LOGGING_DIR = os.path.join("output", FILENAME, "log")
CHECKPOINT_DIR = os.path.join("output", FILENAME, "checkpoints")

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
GAMMA = 0.99
DECAY = 0.995
MINIMAL_SAMPLES = 10000
MAXIMAL_SAMPLES = 50000
ITERATIONS = 5000
DEMO_NUMBER = 10

FRAME_WIDTH = 150
FRAME_HEIGHT = 180
STACK_SIZE = 8

EPSILON_MAX = 1.00
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.99

RENDER_EVERY = 10
SAVE_CHECKPOINT_EVERY = 50

env = EpisodicLifeEnv(gym.make("SpaceInvadersNoFrameskip-v4"))
test_env = gym.wrappers.Monitor(gym.make("SpaceInvadersNoFrameskip-v4"), MONITOR_DIR)

# Image preprocessing
class ImagePreprocessor:
    def __init__(self):
        with tf.variable_scope("image_preprocessor"):
            self.input = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            t = tf.image.convert_image_dtype(self.input, dtype=tf.float32)
            t = tf.image.rgb_to_grayscale(t)
            t = tf.image.crop_to_bounding_box(t, 20, 5, 180, 150)
            self.output = tf.squeeze(t)

    def transform(self, frame, session=None):
        session = session if session is not None else tf.get_default_session()
        return session.run(self.output, feed_dict={self.input: frame})


# Layer Definitions
class ConvLayer(tf.layers.Layer):
    def __init__(self, activation=tf.nn.relu):
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

    def collect_variables(self):
        variables = []
        for layer in [self.conv1, self.conv2, self.conv3]:
            variables += layer.variables
        return variables

    def setup_tensorboard(self):
        variables = self.collect_variables()
        for v in variables:
            tf.summary.histogram(v.name, v)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return x


class QLayer(tf.layers.Layer):
    def __init__(self, output_dim, activation=tf.nn.relu, trainable=True):
        super(QLayer, self).__init__()
        self.W = tf.layers.Dense(
            units=512, activation=activation, trainable=trainable, name="W"
        )
        self.Q = tf.layers.Dense(units=output_dim, trainable=trainable, name="Q")

    def collect_variables(self):
        variables = []
        for layer in [self.W, self.Q]:
            variables += layer.variables
        return variables

    def copy_from(self, other_qlayer):
        assert isinstance(other_qlayer, QLayer)
        target_variables = self.collect_variables()
        source_variables = other_qlayer.collect_variables()
        copy_op = tf.group(
            [
                tf.assign(v_tgt, v_src)
                for v_tgt, v_src in zip(target_variables, source_variables)
            ]
        )
        return copy_op

    def update_from(self, other_qlayer, decay):
        target_variables = self.collect_variables()
        source_variables = other_qlayer.collect_variables()
        update_op = tf.group(
            [
                tf.assign(v_tgt, decay * v_tgt + (1 - decay) * v_src)
                for v_tgt, v_src in zip(target_variables, source_variables)
            ]
        )
        return update_op

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
        return equal_op

    def setup_tensorboard(self):
        variables = self.collect_variables()
        for v in variables:
            tf.summary.histogram(v.name, v)

    def call(self, inputs):
        x = self.W(inputs)
        x = self.Q(x)
        return x


tf.reset_default_graph()

image_preprocessor = ImagePreprocessor()

# Inputs
X = tf.placeholder(
    shape=(None, FRAME_HEIGHT, FRAME_WIDTH, STACK_SIZE), dtype=tf.float32, name="x"
)
X2 = tf.placeholder(
    shape=(None, FRAME_HEIGHT, FRAME_WIDTH, STACK_SIZE), dtype=tf.float32, name="x2"
)
R = tf.placeholder(dtype=tf.float32, shape=(None,), name="reward")  # reward
D = tf.placeholder(dtype=tf.float32, shape=(None,), name="done")  # done
A = tf.placeholder(dtype=tf.int32, shape=(None,), name="action")  # action_dim

# Convolution
conv_layer = ConvLayer()
Z = conv_layer(X)
Z2 = conv_layer(X2)

# Deep Q Network
q_layer = QLayer(output_dim=env.action_space.n, trainable=True)
target_q_layer = QLayer(output_dim=env.action_space.n, trainable=False)

Q = q_layer(Z)
Q2 = target_q_layer(Z2)

with tf.name_scope("q_max"):
    q_avg_max = tf.reduce_mean(tf.reduce_max(Q, axis=1))

# Create ops
with tf.name_scope("predict_op"):
    predict_op = tf.squeeze(tf.argmax(Q, axis=1))

with tf.name_scope("train_op"):
    selected_Q = tf.reduce_sum(
        Q * tf.one_hot(A, env.action_space.n), reduction_indices=[1]
    )
    next_Q = tf.math.reduce_max(Q2, axis=1)
    G = tf.stop_gradient(R + GAMMA * next_Q * (1 - D))
    q_loss = tf.reduce_sum(tf.square(selected_Q - G))
    tf.summary.scalar("QLoss", q_loss)
    train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(q_loss)

with tf.name_scope("copy_op"):
    copy_op = target_q_layer.copy_from(q_layer)

with tf.name_scope("update_op"):
    update_op = target_q_layer.update_from(q_layer, decay=DECAY)

# Initialize Session and Run copy ops
session = tf.Session()
init_op = tf.global_variables_initializer()
session.run(init_op)
session.run(copy_op)

# Setup Summary
conv_layer.setup_tensorboard()
q_layer.setup_tensorboard()
target_q_layer.setup_tensorboard()
summary_op = tf.summary.merge_all()

# This is computed using a different op
q_max_summary_op = tf.summary.scalar("QMax", q_avg_max)

writer = tf.summary.FileWriter(LOGGING_DIR)
writer.add_graph(session.graph)

# Saver
saver = tf.train.Saver(max_to_keep=10)
last_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
if last_checkpoint is not None:
    saver.restore(session, last_checkpoint)
    print("Restored last checkpoint", last_checkpoint)


# Frame Stack
class FrameStack:
    def __init__(self, initial_frame, stack_size=STACK_SIZE):
        self.stack = np.stack([initial_frame] * STACK_SIZE, axis=2)

    def append(self, frame):
        np.append(self.stack[:, :, 1:], np.expand_dims(frame, 2), axis=2)

    def get_state(self):
        return self.stack


replay_buffer = StackedFrameReplayBuffer(
    frame_height=FRAME_HEIGHT,
    frame_width=FRAME_WIDTH,
    stack_size=STACK_SIZE,
    action_dim=1,
    batch_size=BATCH_SIZE,
    max_size=MAXIMAL_SAMPLES,
)

# Play
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

        replay_buffer.store(frame, action, reward, done)

        steps += 1
        total_return += reward
        if render:
            env.render()
    return steps, total_return


# Train
def train(steps):
    for n in range(steps):
        batch = replay_buffer.sample_batch()
        feed_dict = {
            X: batch["s"],
            X2: batch["s2"],
            A: np.squeeze(batch["a"]),
            R: batch["r"],
            D: batch["d"],
        }
        session.run(train_op, feed_dict=feed_dict)
        session.run(update_op)
    return session.run(summary_op, feed_dict)

# Populate replay buffer
print("Populating replay buffer...")
while MINIMAL_SAMPLES > replay_buffer.number_of_samples():
    steps, total_return = play_once(env, 0.0, render=False)
    print("Played %d steps" % steps)

# Collect State Samples
sampled_states = replay_buffer.sample_batch()["s"]

# Main loop
print("Start Main Loop...")
epsilon = EPSILON_MAX
for n in range(ITERATIONS):
    steps, total_return = play_once(env, epsilon, render=(n % RENDER_EVERY == 0))
    t0 = datetime.now()
    train_summary = train(steps)
    delta = datetime.now() - t0
    play_summary = tf.Summary(
        value=[
            tf.Summary.Value(tag="Return", simple_value=total_return),
            tf.Summary.Value(tag="Steps", simple_value=steps),
            tf.Summary.Value(tag="Duration", simple_value=delta.total_seconds()),
            tf.Summary.Value(tag="Epsilon", simple_value=epsilon),
        ]
    )
    q_summary = session.run(q_max_summary_op, feed_dict={
        X: atleast_4d(sampled_states)
    })
    writer.add_summary(play_summary, n)
    writer.add_summary(train_summary, n)
    writer.add_summary(q_summary, n)

    print(
        "Episode:",
        n,
        "Return:",
        total_return,
        "Step:",
        steps,
        "Duration:",
        delta.total_seconds(),
        "Epsilon",
        epsilon,
    )
    if n % SAVE_CHECKPOINT_EVERY == 0:
        path = saver.save(
            session, os.path.join(CHECKPOINT_DIR, "model"), global_step=n
        )
        print("Saved checkpoint to", path)

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

# Demo
for n in range(DEMO_NUMBER):
    play_once(test_env, 0.0, render=True)

# Close Environment
env.close()
