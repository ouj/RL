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
    def __init__(self, state_dim, action_space):
        self.state_dim = state_dim
        self.action_space = action_space
        action_dim = self.action_space.shape[0]

        initializer = tf.keras.initializers.random_normal(mean=0.0, stddev=1.0)
        inputs = tf.keras.layers.Input(shape=(state_dim,))
        dense1 = tf.keras.layers.Dense(
            units=100,
            kernel_initializer=initializer,
            activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dense(
            units=action_dim,
            kernel_initializer=initializer,
            activation=tf.nn.tanh)(dense1)
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


class Agent:

    INITIAL_EXPLORATION = 1.0
    FINAL_EXPLORATION = 0.0
    EXPLORATION_DEC_STEPS = 1000000

    def __init__(
        self,
        env=gym.make('BipedalWalker-v2'),
        population_size=50,
        sigma=0.1,
        learning_rate=0.01,
        decay = 0.999,
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.population_size = population_size
        self.decay = decay
        self.exploration = self.INITIAL_EXPLORATION

    def play_episode(self, model, render=False):
        observation = self.env.reset()
        done = False
        episode_return = 0
        while not done:
            action = self.get_action(model, observation)
            observation, reward, done, _ = self.env.step(action)
            episode_return += reward
            if render: self.env.render()
        return episode_return

    def get_action(self, model, state):
        self.exploration = max(
            self.FINAL_EXPLORATION,
            self.exploration - self.INITIAL_EXPLORATION / self.EXPLORATION_DEC_STEPS)
        if np.random.random() < self.exploration:
            action = self.env.action_space.sample()
        else:
            action = model.predict(state)
        return action

    def train(self, iterations=100):
        model = EvoModel(self.env.observation_space.shape[0], self.env.action_space)
        weights = self.model.get_1d_weights()

        iteration_reward = np.zeros(iterations)
        for t in range(iterations):
            t0 = datetime.now()

            R = np.zeros(self.population_size) # episode return
            N = np.random.randn(self.population_size, len(weights))

            for p in range(self.population_size):
                new_weights = self.weights + N[p] * self.sigma
                model.set_1d_weights(new_weights)
                episode_return = self.play_episode(render=False)
                R[p] = episode_return


            A = (R - R.mean()) / R.std()
            iteration_reward[t] = R.mean()
            print ("Iteration reward:", iteration_reward[t])

            weights = weights + self.learning_rate / (self.population_size * self.sigma) * np.dot(R.T, A)

            # update the learning rate
            self.learning_rate *= self.decay
            print("Iter:", t, "Avg Reward: %.3f" % R.mean(), "Max:", R.max(), "Duration:", (datetime.now() - t0))

            if t != 0 and t % 10 == 0:
                self.model.set_1d_weights(self.weights)
                episode_return, episode_length = self.play_episode(render=True)

        def save_weight(weights, filename="evolution.npy"):
            np.savetxt(filename, weights)

        def load_weight(filename="evolution.npy"):
            if not os.path.exists(filename):
                return None
            return np.load(filename)


def main():
    set_random_seed(0)
    env = gym.make("BipedalWalker-v2")
    agent = Agent(env)
    agent.train()



if __name__ == "__main__":
    main()
