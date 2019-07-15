#!/usr/bin/env python3

import gym
import tensorflow as tf
import os
import numpy as np
from datetime import datetime


def set_random_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)


def save_model(model, filename="evolution.npz"):
    print("Saved model to", filename)
    np.savez(filename, *model)


def load_model(filename="evolution.npz"):
    if not os.path.exists(filename):
        return None
    npzfile = np.load(filename, allow_pickle=True)
    print("Loded model from", filename)
    return [v for _, v in sorted(npzfile.items())]


class Agent:
    def __init__(
        self,
        env=gym.make('BipedalWalker-v2'),
        population_size=50,
        sigma=0.1,
        learning_rate=0.03,
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.population_size = population_size

    def create_network_model(self, state_dim, action_dim, layer_sizes=(100, )):
        model = []
        dim = state_dim
        for h in layer_sizes:
            model.append(np.random.randn(dim, h) / np.sqrt(dim))
            dim = h
        model.append(np.random.randn(dim, action_dim) / np.sqrt(action_dim))
        return model

    def predict(self, model, state):
        x = state
        for w in model:
            x = np.tanh(np.matmul(x, w))
        return x

    def play_episode(self, model, episode_iterations, render=False):
        observation = self.env.reset()
        done = False
        episode_return = 0
        for _ in range(episode_iterations):
            action = self.get_action(model, observation)
            observation, reward, done, _ = self.env.step(action)
            episode_return += reward
            if render:
                self.env.render()
            if done:
                break
        return episode_return

    def get_action(self, model, state):
        return self.predict(model, state)

    def explore(self, model, episode_iterations):
        N = []  # mutations
        R = np.zeros(self.population_size)  # returns
        for v in model:
            N.append(np.random.randn(
                self.population_size, v.shape[0], v.shape[1]))

        for p in range(self.population_size):
            new_model = []
            for v, n in zip(model, N):
                new_model.append(v + self.sigma * n[p])

            R[p] = self.play_episode(
                new_model, episode_iterations, render=False
            )

        A = (R - np.mean(R)) / np.std(R)
        for k in range(len(model)):
            model[k] = model[k] + self.learning_rate / \
                (self.population_size * self.sigma) * \
                np.dot(N[k].transpose(1, 2, 0), A)

        return model

    def train(self, iterations=1000, model=None):
        if model is None:
            model = self.create_network_model(
                self.env.observation_space.shape[0],
                self.env.action_space.shape[0],
                layer_sizes=(100,)
            )

        episode_iterations = 1600
        print("Episode lenght:", episode_iterations)
        avg_reward = None
        for t in range(iterations):
            t0 = datetime.now()
            model = self.explore(model, episode_iterations)
            reward = self.play_episode(
                model, episode_iterations, render=(t % 10 == 0)
            )
            if avg_reward is None:
                avg_reward = reward
            else:
                avg_reward = avg_reward * 0.9 + reward * 0.1

            print(
                "Iteration:", t,
                "Reward:", reward,
                "Avg Reward: %.2f" % avg_reward,
                "Duration:", (datetime.now() - t0)
            )
            if t % 50 == 0:
                save_model(model)
        return model


def main():
    set_random_seed(0)
    env = gym.make("BipedalWalker-v2")
    agent = Agent(env)
    model = load_model()
    model = agent.train(iterations=1000, model=model)
    save_model(model)


if __name__ == "__main__":
    main()
