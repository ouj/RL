#!/usr/bin/env python3.7

import retro
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from rl.helpers import set_random_seed
from rl.mlp import MLPNetwork, MLPSequence
from matplotlib import pyplot as plt

# Set random seeds
set_random_seed(0)

# Path and folders
FILENAME = "kungfu_vpg"
TS = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
MONITOR_DIR = os.path.join("output", FILENAME, "video", TS)
LOGGING_DIR = os.path.join("output", FILENAME, "log", "run1")
CHECKPOINT_DIR = os.path.join("output", FILENAME, "checkpoints")

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

observation = env.reset()
plt.imshow(observation)
frame = image_processor.transform(observation)
frame.shape
plt.imshow(np.squeeze(image_processor.transform(observation)), cmap="gray")

X = tf.placeholder(tf.float32, shape=(None, 80, 240, 1), name='X')
Y = tf.placeholder(tf.float32, shape=(None,), name='Y')
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
    ])

def create_q_net(output_dim, activation=tf.nn.relu, output_activation=tf.nn.softmax, trainable=True):
    return MLPNetwork([
        tf.layers.Flatten(name="flatten"),
        tf.layers.Dense(
            units=512, activation=activation, trainable=trainable, name="W",
            kernel_initializer=tf.initializers.glorot_uniform,
        ),
        tf.layers.Dense(
            units=output_dim, trainable=trainable, name="Q",
            activation=output_activation,
            kernel_initializer=tf.initializers.glorot_uniform
        )
    ])

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
            name="squeeze"),
        ])

conv_net = create_conv_net()
q_net = create_q_net(env.action_space.n)
v_net = create_v_net(env.action_space.n)

Z = conv_net(X)
Q = q_net(Z)
V = v_net(Z)

with tf.name_scope("predict_op"):
    predict_op = tf.squeeze(tf.argmax(Q, axis=1))

with tf.name_scope("q_train_op"):
    selected_Q = tf.reduce_sum(
        Q * tf.one_hot(A, env.action_space.n), reduction_indices=[1]
    )

with tf.name_scope("v_train_op"):
    pass

# Initialize variables
session.run(tf.global_variables_initializer())

def play_once(env, render=False):
    observation = env.reset()
    done = False
    steps = 0
    total_return = 0
    while not done:
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        steps += 1
        total_return += reward
        if render:
            env.render()
    return steps, total_return

play_once(env, render=True)

env.close()
