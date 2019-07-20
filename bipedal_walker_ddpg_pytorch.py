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
BATCH_SIZE = 64

MAX_EPISODE_LENGTH = 1600

SAVE_CHECKPOINT_EVERY = 100
DEMO_EVERY = 100

# Environment
env = gym.make("BipedalWalker-v2")

# Computational Graph
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_mu_net(input_dim, output_dim):
    model = nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, output_dim),
        nn.Tanh()
    )

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)
    model.to(device=device)
    return model


def create_q_net(input_dim):
    model = nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 1),
    )

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)
    model.to(device=device)
    return model


def copy_weight(target, source):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(s.data)


def update_weights(target, source, decay):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(decay * t.data + (1 - decay) * s.data)


observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_max = env.action_space.high[0]

mu_net = create_mu_net(
    input_dim=observation_dim,
    output_dim=action_dim
)
targ_mu_net = create_mu_net(
    input_dim=observation_dim,
    output_dim=action_dim
)
copy_weight(targ_mu_net, mu_net)

q_net = create_q_net(input_dim=observation_dim + action_dim)
targ_q_net = create_q_net(input_dim=observation_dim + action_dim)
copy_weight(targ_q_net, q_net)

mu_optimizer = optim.Adam(mu_net.parameters(), lr=MU_LEARNING_RATE)
q_optimizer = optim.Adam(q_net.parameters(), lr=Q_LEARNING_RATE)


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
    observation_dim,
    action_dim,
    max_size=MAXIMAL_SAMPLES
)


# Play
def get_action(observation):
    with torch.no_grad():
        state = torch.tensor(
            observation, device=device, dtype=torch.float
        ).unsqueeze(dim=0)
        action = action_max * mu_net(state)
        action = action.squeeze().detach().cpu().numpy()
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
        batch = replay_buffer.sample_batch(BATCH_SIZE)

        x = torch.tensor(batch['s'], device=device, dtype=torch.float)
        x2 = torch.tensor(batch['s2'], device=device, dtype=torch.float)
        action = torch.tensor(batch['a'], device=device, dtype=torch.float)
        reward = torch.tensor(
            batch['r'], device=device, dtype=torch.float).unsqueeze(1)
        done = torch.tensor(batch['d'], device=device,
                            dtype=torch.float).unsqueeze(1)

        # Train Q
        q = q_net(torch.cat((x, action), dim=1))
        with torch.no_grad():
            next_mu = action_max * targ_mu_net(x2)
            next_q = targ_q_net(torch.cat((x2, next_mu), dim=1))
            g = reward + GAMMA * (1 - done) * next_q

        q_loss = F.mse_loss(q, g)
        q_optimizer.zero_grad()
        q_loss.backward()
        q_optimizer.step()

        # Train Mu
        mu = action_max * mu_net(x)
        q_mu = q_net(torch.cat((x, mu), dim=1))
        mu_loss = -q_mu.mean()
        mu_optimizer.zero_grad()
        mu_loss.backward()
        mu_optimizer.step()

        update_weights(targ_mu_net, mu_net, DECAY)
        update_weights(targ_q_net, q_net, DECAY)


def demo():
    demo_env = gym.wrappers.Monitor(
        env, MONITOR_DIR, resume=True, mode="evaluation", write_upon_reset=True
    )
    steps, total_return = play_once(demo_env, random_action=False, render=True)
    print("Demo for %d steps, Return %d" % (steps, total_return))
    demo_env.close()


# Populate replay buffer
print("Populating replay buffer")
while MINIMAL_SAMPLES > replay_buffer.number_of_samples():
    steps, total_return = play_once(env, random_action=True, render=False)
    print("Played %d < %d steps" %
          (replay_buffer.number_of_samples(), MINIMAL_SAMPLES))

# Main loop
print("Start Main Loop...")
for n in range(ITERATIONS):
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
    )
    if n % DEMO_EVERY == 0:
        demo()
# Close Environment
env.close()
