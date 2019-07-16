#!/usr/bin/env python3.7
import os
from datetime import datetime
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from common.replay_buffer import ReplayBuffer
from common.helpers import set_random_seed
from common.schedules import LinearSchedule
import matplotlib.pyplot as plt

# Set random seeds
set_random_seed(0)

# Path and folders
FILENAME = "cart_pole_dqn_pytorch"
TS = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
MONITOR_DIR = os.path.join("output", FILENAME, "video", TS)
LOGGING_DIR = os.path.join("output", FILENAME, "log", "run1")
CHECKPOINT_DIR = os.path.join("output", FILENAME, "checkpoints")

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
GAMMA = 0.99
DECAY = 0.99
MINIMAL_SAMPLES = 10000
MAXIMAL_SAMPLES = 1000000
ITERATIONS = 20000

UPDATE_TARGET_EVERY = 10
DEMO_EVERY = 100

# Environment
env = gym.make("CartPole-v1")

global_step = 0

observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Computational Graph
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        hidden_size = 20
        self.w = nn.Linear(in_features=input_dim, out_features=hidden_size)
        self.v = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.q = nn.Linear(in_features=hidden_size, out_features=output_dim)
        self.to(device=device)

    def forward(self, x):
        x = F.relu(self.w(x))
        x = F.relu(self.v(x))
        x = self.q(x)
        return x


policy_net = PolicyNet(observation_dim, action_dim)
target_net = PolicyNet(observation_dim, action_dim)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

# Replay Buffer
replay_buffer = ReplayBuffer(
    observation_shape=env.observation_space.shape,
    action_shape=(1, )
)

# Play Episode


def sample_action(env, observation, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state = torch.tensor(observation, device=device, dtype=torch.float)
            index = policy_net(state).max(0).indices.item()
            return index


def play_once(env, epsilon, render=False):
    observation = env.reset()
    done = False
    steps = 0
    total_return = 0
    while not done:
        action = sample_action(env, observation, epsilon)
        next_observation, reward, done, _ = env.step(action)
        replay_buffer.store(observation, action, reward,
                            next_observation, done)
        observation = next_observation
        steps += 1
        total_return += reward
        if render:
            env.render()
    return steps, total_return

# Train


def train():
    batch = replay_buffer.sample_batch(BATCH_SIZE)
    with torch.no_grad():
        x = torch.tensor(batch['s'], device=device, dtype=torch.float)
        x2 = torch.tensor(batch['s2'], device=device, dtype=torch.float)
        action = torch.tensor(batch['a'], device=device, dtype=torch.long)
        reward = torch.tensor(batch['r'], device=device, dtype=torch.float)
        done = torch.tensor(batch['d'], device=device, dtype=torch.float)

    selected_q = policy_net(x).gather(1, action)
    with torch.no_grad():
        next_q = target_net(x2).max(1).values
        g = reward + GAMMA * next_q * (1 - done)

    # Compute Huber loss
    loss = F.mse_loss(selected_q, g.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    # for param in target_net.parameters():
    #     print(param)
    optimizer.step()


def demo():
    demo_env = gym.wrappers.Monitor(
        env,
        MONITOR_DIR,
        resume=True,
        mode="evaluation",
        write_upon_reset=True
    )
    steps, total_return = play_once(demo_env, 0.05, render=True)
    print("Demo for %d steps, Return %d" % (steps, total_return))


# Populate replay buffer
epsilon = 1.0
print("Populating replay buffer with epsilon %f..." % epsilon)
while MINIMAL_SAMPLES > replay_buffer.number_of_samples():
    steps, total_return = play_once(env, epsilon, render=False)
    print(
        "Played %d < %d steps" %
        (replay_buffer.number_of_samples(), MINIMAL_SAMPLES))


# Main loop
print("Start Main Loop...")
n = 0.5
for n in range(ITERATIONS):
    epsilon = 1.0 / np.sqrt(n+1)
    steps, total_return = play_once(env, epsilon)

    t0 = datetime.now()
    for _ in range(steps):
        train()
        global_step += 1
        if global_step != 0 and global_step % UPDATE_TARGET_EVERY == 0:
            target_net.load_state_dict(policy_net.state_dict())
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
        global_step
    )

    if n % DEMO_EVERY == 0:
        demo()

# Close Environment
env.close()
