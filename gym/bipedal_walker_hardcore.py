#!/usr/bin/env python3

import gym
import tensorflow as tf
import os
import numpy as np
from datetime import datetime

def set_random_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)


def update_weights(weights, update, factor):
    new_weights = []
    for w1, w2 in zip(weights, update):
        new_weights.append(w1 + w2 * factor)
    return new_weights

def mutate_weights(weights, sigma):
    mutation = np.random.randn(len(weights))
    return weights + mutation * sigma

class EvoModel:
    def __init__(self, state_dim, action_space):
        self.state_dim = state_dim
        self.action_space = action_space
        action_dim = self.action_space.shape[0]

        initializer = tf.keras.initializers.random_normal(mean=0.0, stddev=10)
        inputs = tf.keras.layers.Input(shape=(state_dim,))
        dense1 = tf.keras.layers.Dense(
            units=200,
            kernel_initializer=initializer,
            activation=tf.nn.relu)(inputs)
        dense2 = tf.keras.layers.Dense(
            units=200,
            kernel_initializer=initializer,
            activation=tf.nn.relu)(dense1)
        outputs = tf.keras.layers.Dense(
            units=action_dim,
            kernel_initializer=initializer,
            activation=tf.nn.tanh)(dense2)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    def get_1d_weights(self):
        return np.concatenate(
            [w.ravel() for w in self.model.get_weights()]
        )

    def set_1d_weights(self, weights):
        new_weights = []
        index = 0
        for w in self.model.get_weights():
            new_weights.append(weights[index:index+w.size].reshape(w.shape))
            index += w.size
        self.model.set_weights(new_weights)

    def predict(self, state):
        action = self.model.predict(np.atleast_2d(state))[0]
        action_min = self.action_space.low
        action_max = self.action_space.high
        action = action_max * action
        return np.clip(action, action_min, action_max)

    def mutate(self, sigma):
        new_model = EvoModel(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
        new_model.set_1d_weights(mutate_weights(self.get_1d_weights(), sigma))
        return new_model


class Agent:

    INITIAL_EXPLORATION = 1.0
    FINAL_EXPLORATION = 0.0
    EXPLORATION_DEC_STEPS = 1000000

    def __init__(
        self,
        env=gym.make('BipedalWalkerHardcore-v2'),
        population_size=50,
        sigma=0.1,
        learning_rate=0.01,
        decay = 0.999,
        stack_size=8
    ):
        self.env = env
        self.stack_size = stack_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.population_size = population_size
        self.decay = decay

        state_dim = self.env.observation_space.shape[0] * self.stack_size
        self.model = EvoModel(state_dim, self.env.action_space)
        self.weights = self.model.get_1d_weights()
        self.exploration = self.INITIAL_EXPLORATION

    def play_episode(self, render=False):
        observation = self.env.reset()
        done = False
        episode_return = 0
        episode_length = 0
        sequence = [observation] * self.stack_size
        while not done:
            action = self.get_action(np.concatenate(sequence))
            observation, reward, done, _ = self.env.step(action)
            episode_return += reward
            episode_length += 1
            if render:
                self.env.render()
            sequence = sequence[1:] + [observation]
        return episode_return, episode_length

    def get_action(self, state):
        self.exploration = max(
            self.FINAL_EXPLORATION,
            self.exploration - self.INITIAL_EXPLORATION / self.EXPLORATION_DEC_STEPS)
        if np.random.random() < self.exploration:
            action = self.env.action_space.sample()
        else:
            action = self.model.predict(state)
        return action


    def train(self, iterations=100):
        iteration_reward = np.zeros(iterations)
        for t in range(iterations):
            t0 = datetime.now()

            returns = np.zeros(self.population_size) # episode return
            mutations = np.random.randn(self.population_size, len(self.weights))

            for p in range(self.population_size):
                new_weights = self.weights + mutations[p] * self.sigma
                self.model.set_1d_weights(new_weights)
                episode_return, episode_length = self.play_episode(render=False)
                returns[p] = episode_return


            m = returns.mean()
            s = returns.std()
            if s == 0:
                continue

            iteration_reward[t] = m
            print ("Iteration reward:", m)
            A = (returns - m) / s

            self.weights = self.weights + self.learning_rate / (self.population_size * self.sigma) * np.dot(mutations.T, A)

            # update the learning rate
            self.learning_rate *= self.decay
            print("Iter:", t, "Avg Reward: %.3f" % m, "Max:", returns.max(), "Duration:", (datetime.now() - t0))

            if t != 0 and t % 10 == 0:
                self.model.set_1d_weights(self.weights)
                episode_return, episode_length = self.play_episode(render=True)

        def save_weight(filename="evolution.npy"):
            np.savetxt(filename, self.weights)

        def load_weight(filename="evolution.npy"):
            if not os.path.exists(filename):
                return None
            self.weights = np.load(filename)


def main():
    set_random_seed(0)
    env = gym.make("BipedalWalkerHardcore-v2")
    agent = Agent(env)
    agent.train()


if __name__ == "__main__":
    main()





