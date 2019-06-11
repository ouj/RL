#!/usr/bin/env python3

import gym
import numpy as np

def get_action(observation, params):
    return 1 if params.dot(observation) > 0 else 0

def play_episode(env, params):
    total_reward = 0
    # observation:
    # 0 - Cart position
    # 1 - Cart Velocity
    # 2 - Pole Angle
    # 3 - Pole Velocity At Tip
    observation = env.reset()
    for _ in range(2000):
        action = get_action(observation, params)
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        if reward == 0 or done:
            break
        env.render()
    return total_reward

def random_params():
    return np.random.random(4) * 2 -1 # generate random parameters

def random_search(env, number_of_episode):
    best_params = None
    best_reward = 0
    for _ in range(number_of_episode):
        params = random_params()
        reward = play_episode(env, params)
        print ("Parameter: %s, Reward: %s" % (params, reward))
        if reward > best_reward:
            best_reward = reward
            best_params = params
    return best_params, best_reward

def main():
    env = gym.make('CartPole-v1')
    params, reward = random_search(env, 100)
    print("Best Paramester: %s, Best Reward: %d" % (params, reward))
    print("Play Final Episode")
    play_episode(env, params)
    env.close()

if __name__ == "__main__":
    main()

