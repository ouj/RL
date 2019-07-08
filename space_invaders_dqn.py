#!/usr/bin/env python3.7
import os
import sys
from collections import deque
from datetime import datetime
import gym
import numpy as np
import tensorflow as tf
from common.helpers import atleast_4d, set_random_seed
from common.stacked_frame_replay_buffer import StackedFrameReplayBuffer
from common.schedules import LinearSchedule
from wrappers.atari_wrappers import EpisodicLifeEnv
from wrappers.atari_wrappers import WarpFrame


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set random seeds
set_random_seed(0)

# Path and folders
FILENAME = "space_invadors_dpn"
TS = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
MONITOR_DIR = os.path.join("output", FILENAME, "video", TS)
LOGGING_DIR = os.path.join("output", FILENAME, "log", "run1")
CHECKPOINT_DIR = os.path.join("output", FILENAME, "checkpoints")

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
GAMMA = 0.99
DECAY = 0.999
MINIMAL_SAMPLES = 10000
MAXIMAL_SAMPLES = 30000
ITERATIONS = 10000


FRAME_WIDTH = 150
FRAME_HEIGHT = 190
STACK_SIZE = 4

EPSILON_MAX = 1.00
EPSILON_MIN = 0.1
EPSILON_STEPS = 5000000

SAVE_CHECKPOINT_EVERY = 100
DEMO_EVERY = 10

env_name = "SpaceInvadersDeterministic-v4"

def make_train_env(env_name):
    e = gym.make(env_name)
    assert e.frameskip == 3
    return e

def make_test_env(env_name):
    e = gym.make(env_name)
    assert e.frameskip == 3
    return e

env = make_train_env(env_name)

# Image preprocessing
class ImagePreprocessor:
    def __init__(self):
        with tf.variable_scope("image_preprocessor"):
            self.input = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            t = tf.image.rgb_to_grayscale(self.input)
            t = tf.image.convert_image_dtype(t, dtype=tf.float32)
            t = tf.image.crop_to_bounding_box(t, 10, 5, 190, 150)
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
            kernel_initializer=tf.initializers.he_normal(),
            name="conv1",
            use_bias=False,
        )
        self.batch_norm1 = tf.layers.BatchNormalization(
            epsilon=1e-5,
            name="batch_norm1",
        )
        self.conv2 = tf.layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=2,
            padding="valid",
            activation=activation,
            kernel_initializer=tf.initializers.he_normal(),
            name="conv2",
            use_bias=False
        )
        self.batch_norm2 = tf.layers.BatchNormalization(
            epsilon=1e-5,
            name="batch_norm2",
        )
        self.conv3 = tf.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding="valid",
            activation=activation,
            kernel_initializer=tf.initializers.he_normal(),
            name="conv3",
            use_bias=True,
        )

    def collect_variables(self):
        variables = []
        for layer in [self.conv1, self.batch_norm1, self.conv2, self.batch_norm2, self.conv3]:
            variables += layer.variables
        return variables

    def setup_tensorboard(self):
        variables = self.collect_variables()
        for v in variables:
            tf.summary.histogram(v.name, v)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.conv3(x)
        return x


class QLayer(tf.layers.Layer):
    def __init__(self, output_dim, activation=tf.nn.relu, trainable=True):
        super(QLayer, self).__init__()
        self.flatten = tf.layers.Flatten(name="flatten")
        self.W = tf.layers.Dense(
            units=512, activation=activation, trainable=trainable, name="W",
            kernel_initializer=tf.initializers.glorot_uniform,
        )
        self.Q = tf.layers.Dense(
            units=output_dim, trainable=trainable, name="Q",
            kernel_initializer=tf.initializers.glorot_uniform
        )

    def collect_variables(self):
        variables = []
        for layer in [self.flatten, self.W, self.Q]:
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
        x = self.flatten(inputs)
        x = self.W(x)
        x = self.Q(x)
        return x


tf.reset_default_graph()

image_preprocessor = ImagePreprocessor()

# Inputs
X = tf.placeholder(
    shape=(
        None,
        FRAME_HEIGHT,
        FRAME_WIDTH,
        STACK_SIZE),
    dtype=tf.float32,
    name="x")
X2 = tf.placeholder(
    shape=(
        None,
        FRAME_HEIGHT,
        FRAME_WIDTH,
        STACK_SIZE),
    dtype=tf.float32,
    name="x2")
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

global_step = tf.train.get_or_create_global_step()

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
    tf.summary.histogram("G", G)
    tf.summary.scalar("G_mean", tf.reduce_mean(G))
    # https://openai.com/blog/openai-baselines-dqn/ suggest huber_loss
    q_loss = tf.reduce_sum(tf.losses.huber_loss(selected_Q, G))
    tf.summary.scalar("QLoss", q_loss)
    train_op = tf.train.AdamOptimizer(
        learning_rate=LEARNING_RATE
    ).minimize(q_loss, global_step=global_step)

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

writer = tf.summary.FileWriter(LOGGING_DIR)
writer.add_graph(session.graph)

# Saver
saver = tf.train.Saver()
last_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
if last_checkpoint is not None:
    saver.restore(session, last_checkpoint)
    print("Restored last checkpoint", last_checkpoint)


# Frame Stack
class FrameStack:
    def __init__(self, initial_frame, stack_size=STACK_SIZE):
        self.stack = deque(maxlen=stack_size)
        for _ in range(stack_size):
            self.stack.append(initial_frame)

    def append(self, frame):
        self.stack.append(frame)

    def get_state(self):
        return np.stack(self.stack, axis=2)


replay_buffer = StackedFrameReplayBuffer(
    frame_height=FRAME_HEIGHT,
    frame_width=FRAME_WIDTH,
    stack_size=STACK_SIZE,
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
            A: batch["a"],
            R: batch["r"],
            D: batch["d"],
        }
        session.run(
            [train_op, update_op], feed_dict=feed_dict)
    return session.run(summary_op, feed_dict)


def demo():
    demo_env = gym.wrappers.Monitor(
        make_test_env(env_name),
        MONITOR_DIR,
        resume=True,
        mode="evaluation",
        write_upon_reset=True
    )
    steps, total_return = play_once(demo_env, 0.05, render=True)
    print("Demo for %d steps, Return %d" % (steps, total_return))
    summary = tf.Summary()
    summary.value.add(tag="demo/return", simple_value=total_return)
    summary.value.add(tag="demo/steps", simple_value=steps)
    demo_env.close()
    return summary


linear_schedule = LinearSchedule(
    int(EPSILON_STEPS),
    final_p=EPSILON_MIN,
    initial_p=EPSILON_MAX
)
epsilon = linear_schedule.value(session.run(global_step))
# Populate replay buffer
print("Populating replay buffer with epsilon %f..." % epsilon)
while MINIMAL_SAMPLES > replay_buffer.number_of_samples():
    steps, total_return = play_once(env, epsilon, render=False)
    print(
        "Played %d < %d steps" %
        (replay_buffer.number_of_samples(), MINIMAL_SAMPLES))

# Main loop
print("Start Main Loop...")
for n in range(ITERATIONS):
    gstep = tf.train.global_step(session, global_step)
    epsilon = linear_schedule.value(gstep)
    steps, total_return = play_once(env, epsilon)
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
        "Epsilon",
        epsilon,
        "Global Steps:",
        gstep
    )

    summary = tf.Summary()
    summary.value.add(tag="misc/return", simple_value=total_return)
    summary.value.add(tag="misc/steps", simple_value=steps)
    summary.value.add(tag="misc/duration", simple_value=delta.total_seconds())
    summary.value.add(tag="misc/epsilon", simple_value=epsilon)
    writer.add_summary(train_summary, global_step=gstep)
    writer.add_summary(summary, global_step=gstep)

    if n != 0 and n % SAVE_CHECKPOINT_EVERY == 0:
        path = saver.save(
            session,
            os.path.join(CHECKPOINT_DIR, "model"),
            global_step=gstep
        )
        print("Saved checkpoint to", path)

    if n % DEMO_EVERY == 0:
        summary = demo()
        writer.add_summary(summary, global_step=gstep)

# Close Environment
env.close()
