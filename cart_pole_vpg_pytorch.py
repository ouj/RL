#!/usr/bin/env python3

import os
from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from common.helpers import set_random_seed
from common.schedules import LinearSchedule

set_random_seed(0)

# Path and folders
FILENAME = "cart_pole_vpg_pytorch"
TS = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
MONITOR_DIR = os.path.join("output", FILENAME, "video", TS)
LOGGING_DIR = os.path.join("output", FILENAME, "log", "run")
CHECKPOINT_DIR = os.path.join("output", FILENAME, "checkpoints")

# Hyperparameters
P_LEARNING_RATE = 1e-3
V_LEARNING_RATE = 1e-3
GAMMA = 0.99
ITERATIONS = 10000
SAVE_CHECKPOINT_EVERY = 100
DEMO_EVERY = 10

SAVE_CHECKPOINT_EVERY = 100
DEMO_EVERY = 100

# Setup
env = gym.make("CartPole-v1")

# Computational Graph
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_policy_nets(input_dim, output_dim):
    model = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, output_dim),
        nn.Softmax(dim=1),
    )
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)
    model.to(device=device)
    return model

def create_value_net(input_dim):
    model = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)
    model.to(device=device)
    return model

policy_net = create_policy_nets(
    env.observation_space.shape[0],
    env.action_space.n
)
value_net = create_value_net(env.observation_space.shape[0], )

policy_optimizer = optim.Adam(policy_net.parameters(), lr=P_LEARNING_RATE)
value_optimizer = optim.Adam(value_net.parameters(), lr=V_LEARNING_RATE)

# X = tf.placeholder(
#     shape=(None, env.observation_space.shape[0]), dtype=tf.float32, name="x"
# )
# A = tf.placeholder(
#     dtype=tf.int32, shape=(None, *env.action_space.shape), name="action"
# )  # action
# G = tf.placeholder(tf.float32, shape=(None,), name='G')




# Q = policy_net(X)
# V = value_net(X)
#
# with tf.name_scope("train_op"):
#     with tf.name_scope("Policy"):
#         advantages = tf.stop_gradient(G - V)
#         selected_prob = tf.log(tf.reduce_sum(
#             Q * tf.one_hot(A, env.action_space.n), reduction_indices=[1]
#         ))
#         q_loss = -tf.reduce_sum(advantages * selected_prob)
#         q_train_op = tf.train.AdamOptimizer(
#             learning_rate=P_LEARNING_RATE).minimize(q_loss)
#         tf.summary.scalar("policy_loss", q_loss)
#         tf.summary.scalar("max_advantages", tf.math.reduce_max(advantages))
#         tf.summary.histogram("advantages", advantages)
#
#     with tf.name_scope("Value"):
#         v_loss = tf.reduce_sum(tf.square(G - V))
#         v_train_op = tf.train.AdamOptimizer(V_LEARNING_RATE).minimize(v_loss)
#         tf.summary.histogram("value", V)
#         tf.summary.scalar("max_value", tf.math.reduce_max(V))

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
    with torch.no_grad():
        state = torch.tensor(
            observation, device=device, dtype=torch.float
        ).unsqueeze(dim=0)
        p = policy_net(state)
        p = p.squeeze().cpu().numpy()
    return np.random.choice(len(p), p=p)

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
    X = torch.tensor(states, device=device, dtype=torch.float)
    A = torch.tensor(actions, device=device, dtype=torch.long).unsqueeze(1)
    G = torch.tensor(returns, device=device, dtype=torch.float).unsqueeze(1)

    # Train policy net
    with torch.no_grad():
        V = value_net(X)
        advantages = G - V
    selected_prob = torch.log(policy_net(X).gather(1, A))
    q_loss = -torch.sum(advantages * selected_prob)
    policy_optimizer.zero_grad()
    q_loss.backward()
    policy_optimizer.step()

    V = value_net(X)
    v_loss = F.mse_loss(G, V)
    value_optimizer.zero_grad()
    v_loss.backward()
    value_optimizer.step()


def demo():
    demo_env = gym.wrappers.Monitor(
        env, MONITOR_DIR, resume=True, mode="evaluation", write_upon_reset=True
    )
    steps, total_return, _, _, _ = play(demo_env, render=True)
    print("Demo for %d steps, Return %d" % (steps, total_return))
    demo_env.close()


print("Start Main Loop...")
for n in range(ITERATIONS):
    total_steps, total_rewards, states, actions, returns = play(
        env, render=False)
    t0 = datetime.now()
    train(states, actions, returns)
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
    )

    if n % DEMO_EVERY == 0:
        demo()

env.close()
