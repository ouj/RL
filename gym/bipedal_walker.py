#!/usr/bin/env python3

import gym
import tensorflow as tf
import numpy as np
from rl.replay_buffer import ReplayBuffer
import os

def set_random_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)

# simple feedforward neural net
def ANN(x, layer_sizes, hidden_activation=tf.nn.relu, output_activation=None):
    for h in layer_sizes[:-1]:
        x = tf.layers.Dense(
            units=h, activation=hidden_activation)(x)
        x = tf.layers.Dropout(0.1)(x)
    return tf.layers.Dense(
        units=layer_sizes[-1], activation=output_activation)(x)


# get all variables within a scope
def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]



### Create both the actor and critic networks at once ###
### Q(s, mu(s)) returns the maximum Q for a given state s ###
def create_networks(
    s, a,
    num_actions,
    action_max,
    action_min,
    hidden_sizes=[300,],
    hidden_activation=tf.nn.relu,
    output_activation=tf.sigmoid):

    with tf.variable_scope('mu'):
        mu = ANN(s, hidden_sizes+[num_actions], hidden_activation, output_activation)
        mu = (action_max - action_min) * mu + action_min
    with tf.variable_scope('q'):
        input_ = tf.concat([s, a], axis=-1) # (state, action)
        q = tf.squeeze(ANN(input_, hidden_sizes+[1], hidden_activation, None), axis=1)
    with tf.variable_scope('q', reuse=True):
        # reuse is True, so it reuses the weights from the previously defined Q network
        input_ = tf.concat([s, mu], axis=-1) # (state, mu(state))
        q_mu = tf.squeeze(ANN(input_, hidden_sizes+[1], hidden_activation, None), axis=1)
    return mu, q, q_mu


class DDPG:
    def __init__(self, env, max_episode_length):
        self.action_noise = 0.1
        self.gamma = 0.999
        self.decay = 0.995
        self.mu_learning_rate=1e-3
        self.q_learning_rate=1e-3

        self.env = env
        self.max_episode_length = max_episode_length
        self.observation_number = env.observation_space.shape[0]
        self.action_number = env.action_space.shape[0]
        self.action_max = env.action_space.high
        self.action_min = env.action_space.low

        self.checkpoint_dir = "checkpoint"


    def init(self, hidden_layers):
        self.replay_buffer = ReplayBuffer(
            observation_dim=self.observation_number,
            action_dim=self.action_number
        )
        self.create_neuro_networks(hidden_layers)
        print ("DDPG model initialized")

    def create_neuro_networks(self, hidden_layers):
        tf.reset_default_graph()
        self.X = tf.placeholder(
            dtype=tf.float32, shape=(None, self.observation_number)
        ) # observation
        self.A = tf.placeholder(
            dtype=tf.float32, shape=(None, self.action_number)
        ) # action
        self.X2 = tf.placeholder(
            dtype=tf.float32, shape=(None, self.observation_number)
        ) # next observation
        self.R = tf.placeholder(dtype=tf.float32, shape=(None,)) # reward
        self.D = tf.placeholder(dtype=tf.float32, shape=(None,)) # done

        # Main network outputs
        with tf.variable_scope('main'):
            self.mu, self.q, self.q_mu = create_networks(
                self.X, self.A,
                hidden_sizes = hidden_layers,
                num_actions=self.action_number,
                action_max=self.action_max,
                action_min=self.action_min
            )

        with tf.variable_scope("target"):
            _, _, self.q_mu_target = create_networks(
                self.X2, self.A,
                hidden_sizes = hidden_layers,
                num_actions=self.action_number,
                action_max=self.action_max,
                action_min=self.action_min
            )

        # Target value for the Q-network loss
        # We use stop_gradient to tell Tensorflow not to differentiate
        # q_mu_targ wrt any params
        # i.e. consider q_mu_targ values constant
        self.q_target = tf.stop_gradient(
            self.R + self.gamma * (1 - self.D) * self.q_mu_target
        )

        # DDPG losses
        self.mu_loss = -tf.reduce_mean(self.q_mu)
        self.q_loss = tf.reduce_mean((self.q - self.q_target)**2)

        # Train each network separately
        mu_optimizer = tf.train.AdamOptimizer(learning_rate=self.mu_learning_rate)
        q_optimizer = tf.train.AdamOptimizer(learning_rate=self.q_learning_rate)
        self.mu_train_op = mu_optimizer.minimize(
            self.mu_loss, var_list=get_vars('main/mu')
        )
        self.q_train_op = q_optimizer.minimize(
            self.q_loss, var_list=get_vars('main/q')
        )

        # Use soft updates to update the target networks
        self.target_update = tf.group([
            tf.assign(v_targ, self.decay * v_targ + (1 - self.decay) * v_main)
            for v_main, v_targ in zip(get_vars('main'), get_vars('target'))
        ])

        # Copy main network params to target networks
        self.target_init = tf.group([
            tf.assign(v_targ, v_main)
            for v_main, v_targ in zip(get_vars('main'), get_vars('target'))
        ])

        # boilerplate (and copy to the target networks!)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.session.run(self.target_init)
        for v in tf.global_variables():
            print(v.name, v.shape)


    def get_action(self, observation, noise_scale):
        action = self.session.run(self.mu, feed_dict={
            self.X: observation.reshape(1,-1)
        })[0]
#        action = np.clip(action, self.action_min, self.action_max)
        action += noise_scale * np.random.randn(self.action_number)
        return np.clip(action, self.action_min, self.action_max)

    def play_episode(self, random_action=False, render=False):
        done = False
        observation = self.env.reset()
        episode_return = 0
        episode_length = 0
        action_noise = self.action_noise
        while not done:
            if random_action:
                action = self.env.action_space.sample()
            else:
                action = self.get_action(observation, action_noise)

            next_observation, reward, done, _ = self.env.step(action)

            if np.sum(np.abs(next_observation - observation)) < 1e-5:
                # if it stucks, we try to explore and get out of the jam
                action_noise += self.action_noise

            episode_return += reward
            episode_length += 1

            self.replay_buffer.store(
                observation, action, reward, next_observation,
                False if episode_length == self.max_episode_length else done
            )
            observation = next_observation
            if render:
                self.env.render()

        return episode_return, episode_length

    def train(
        self,
        episode_number=200,
        random_episodes=25,
        test_agent_every=25,
        save_checkpoint_every=50,
    ):
        q_losses = []
        mu_losses = []
        episode_returns = []

        for e in range(episode_number):
            if e == random_episodes:
                print ("=== START PLAYING USING AGENT ===")

            episode_return, episode_length = self.play_episode(
                random_action=(e < random_episodes)
            )
            episode_returns.append(episode_return)

            if self.replay_buffer.number_of_samples() > 10000:
                ql, mul = self.update(episode_length)
                q_losses += ql
                mu_losses += mul

                print("Played Episode %d, Return %d, Length %d, Q loss %f, Mu loss %f" % (
                    e, episode_return, episode_length, np.mean(ql), np.mean(mul)
                ))
            else:
                print("Played Episode %d, Return %d, Length %d, Replay Buffer Size %d" % (
                    e, episode_return, episode_length, self.replay_buffer.number_of_samples()
                ))

            if e > 0 and e % test_agent_every == 0:
                self.test()

            if e > 0 and e % save_checkpoint_every == 0:
                self.save_checkpoint(global_step=e)

        return episode_returns, q_losses, mu_losses



    def update(self, episode_length, batch_size=64):
        q_losses = []
        mu_losses = []
        for _ in range(episode_length):
            batch = self.replay_buffer.sample_batch(batch_size)
            feed_dict = {
                self.X: batch['s'],
                self.X2: batch['s2'],
                self.A: batch['a'],
                self.R: batch['r'],
                self.D: batch['d']
            }
            ql, _, _ = self.session.run(
                [self.q_loss, self.q, self.q_train_op], feed_dict
            )
            mul, _, _ = self.session.run(
                [self.mu_loss, self.mu_train_op, self.target_update], feed_dict
            )
            q_losses.append(ql)
            mu_losses.append(mul)
        return q_losses, mu_losses

    def test(self, episode_number=1):
        for _ in range(episode_number):
            done = False
            observation = self.env.reset()
            episode_return = 0
            episode_length = 0
            while not done:
                action = self.get_action(observation, self.action_noise)
                next_observation, reward, done, _ = self.env.step(action)

                episode_return += reward
                episode_length += 1
                observation = next_observation
                self.env.render()
            print(
                'Test return:', episode_return,
                'Episode Length:', episode_length
            )


    def save_checkpoint(self, global_step=None):
        model_file = os.path.join(self.checkpoint_dir, "model")
        os.makedirs(os.path.basename(model_file), exist_ok=True)
        save_path = self.saver.save(
            self.session,
            global_step=global_step,
            save_path=model_file)
        print ("Saved checkpoint to", save_path)


    def restore_checkpoint(self):
        model_file = os.path.join(self.checkpoint_dir, "model")
        try:
            self.saver.restore(self.session, model_file)
            print ("Restore checkpoint to", model_file)
        except:
            print ("Failed to load checkpoint")


    def cleanup(self):
        tf.reset_default_graph()
        self.env.close()

def play3(env):
    ddpg = DDPG(env, max_episode_length=1600)
    ddpg.init(hidden_layers=[200, 200])
    ddpg.restore_checkpoint()
    for _ in range(10):
        episode_return, episode_length = ddpg.play_episode(
            random_action=False, render=True
        )
        print ("Episode Return", episode_return, "Episode Length", episode_length)
    ddpg.cleanup()

def train3(env):
    ddpg = DDPG(env, max_episode_length=1600)
    ddpg.init(hidden_layers=[200, 200])
    ddpg.restore_checkpoint()
    episode_returns, q_losses, mu_losses = ddpg.train(
        episode_number=15000,
        random_episodes=50,
        save_checkpoint_every=50
    )
    ddpg.save_checkpoint()
    ddpg.cleanup()

def main():
    set_random_seed(0)
    env = gym.make("BipedalWalker-v2")
#    play3(env)
    train3(env)
    train3(env)


if __name__ == "__main__":
    main()

