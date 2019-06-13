#!/usr/bin/env python3

import gym
import tensorflow as tf
import os
import numpy as np
from datetime import datetime

def set_random_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)

class EvoModel:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim

#        initializer = tf.keras.initializers.random_normal(mean=0.0, stddev=1.0)
#        inputs = tf.keras.layers.Input(shape=(state_dim,))
#        dense1 = tf.keras.layers.Dense(
#            units=100,
#            kernel_initializer=initializer,
#            activation=tf.nn.relu)(inputs)
#        outputs = tf.keras.layers.Dense(
#            units=action_dim,
#            kernel_initializer=initializer,
#            activation=tf.nn.tanh)(dense1)
#        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        hl_size = 100
        self.model = []
        self.model.append(np.random.randn(24, hl_size) / np.sqrt(24))
        self.model.append(np.random.randn(hl_size, 4) / np.sqrt(hl_size))

    def get_weights(self):
        return self.model

    def set_weights(self, weights):
        self.model = weights

    def predict(self, state):
#        action = self.model.predict(np.atleast_2d(state))[0]
        hl = np.matmul(state, self.model[0])
        hl = np.tanh(hl)
        action = np.matmul(hl, self.model[1])
        action = np.tanh(action)
        return action


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

    def play_episode(self, model, max_iterations=300, render=False):
        observation = self.env.reset()
        done = False
        episode_return = 0
        for _ in range(max_iterations):
            action = self.get_action(model, observation)
            observation, reward, done, _ = self.env.step(action)
            episode_return += reward
            if render:
                self.env.render()
            if done:
                break
        return episode_return

    def get_action(self, model, state):
        return model.predict(state)

    def explore(self, model):
        weights = model.get_weights()

        N = [] # mutations
        R = np.zeros(self.population_size) # returns
        for v in weights:
            N.append(np.random.randn(self.population_size, v.shape[0], v.shape[1]))

        for p in range(self.population_size):
            weights_try = []
            for v, n in zip(weights, N):
                weights_try.append(v + self.sigma * n[p])

            model.set_weights(weights_try)
            R[p] = self.play_episode(model, render=False)

        A = (R - np.mean(R)) / np.std(R)
        for k in range(len(weights)):
            weights[k] = weights[k] + self.learning_rate/(self.population_size * self.sigma) * np.dot(N[k].transpose(1, 2, 0), A)
        model.set_weights(weights)

    def train(self, iterations=100):
        model = EvoModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0]
        )
        avg_reward = None
        for t in range(iterations):
            t0 = datetime.now()
            self.explore(model)
            reward = self.play_episode(model, render=(t % 10 == 0))
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



    def save_weights(self, weights, filename="evolution.npy"):
        np.savetxt(filename, weights)

    def load_weights(self, filename="evolution.npy"):
        if not os.path.exists(filename):
            return None
        return np.load(filename)


def main():
    set_random_seed(0)
    env = gym.make("BipedalWalker-v2")
    agent = Agent(env)
    agent.train(400)



if __name__ == "__main__":
    main()
