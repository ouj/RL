#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join("..",".."))

import gym
import tensorflow as tf
import numpy as np
from libs.rl.replay_buffer import ReplayBuffer

# simple feedforward neural net
def ANN(x, layer_sizes, hidden_activation=tf.nn.relu, output_activation=None):
    for h in layer_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=hidden_activation)
    return tf.layers.dense(x, units=layer_sizes[-1], activation=output_activation)


# get all variables within a scope
def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]


### Create both the actor and critic networks at once ###
### Q(s, mu(s)) returns the maximum Q for a given state s ###
def create_networks(
    s, a,
    num_actions,
    action_max,
    hidden_sizes=(300,),
    hidden_activation=tf.nn.relu,
    output_activation=tf.tanh):

    with tf.variable_scope('mu'):
        mu = action_max * ANN(s, list(hidden_sizes)+[num_actions], hidden_activation, output_activation)
    with tf.variable_scope('q'):
        input_ = tf.concat([s, a], axis=-1) # (state, action)
        q = tf.squeeze(ANN(input_, list(hidden_sizes)+[1], hidden_activation, None), axis=1)
    with tf.variable_scope('q', reuse=True):
        # reuse is True, so it reuses the weights from the previously defined Q network
        input_ = tf.concat([s, mu], axis=-1) # (state, mu(state))
        q_mu = tf.squeeze(ANN(input_, list(hidden_sizes)+[1], hidden_activation, None), axis=1)
    return mu, q, q_mu


class DDPG:
    def __init__(
        self,
        env,
        random_seed=0,
        gamma=0.99,
        decay=0.995,
        mu_learning_rate=1e-4,
        q_learning_rate=1e-4,
        action_noise=0.05,
        checkpoint_dir="checkpoint"
    ):
        self.env = env
        tf.reset_default_graph()

        tf.set_random_seed(random_seed)
        np.random.seed(random_seed)

        self.observation_number = env.observation_space.shape[0]
        self.action_number = env.action_space.shape[0]
        self.action_max = env.action_space.high[0]
        self.action_noise = action_noise
        self.checkpoint_dir = checkpoint_dir

        self.replay_buffer = ReplayBuffer(
            observation_dim=self.observation_number,
            action_dim=self.action_number
        )

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
                hidden_sizes=[50, 50],
                num_actions=self.action_number,
                action_max=self.action_max
            )

        with tf.variable_scope("target"):
            _, _, self.q_mu_target = create_networks(
                self.X2, self.A,
                hidden_sizes=[50, 50],
                num_actions=self.action_number,
                action_max=self.action_max
            )

        # Target value for the Q-network loss
        # We use stop_gradient to tell Tensorflow not to differentiate
        # q_mu_targ wrt any params
        # i.e. consider q_mu_targ values constant
        self.q_target = tf.stop_gradient(
            self.R + gamma * (1 - self.D) * self.q_mu_target
        )

        # DDPG losses
        self.mu_loss = -tf.reduce_mean(self.q_mu)
        self.q_loss = tf.reduce_mean((self.q - self.q_target)**2)

        # Train each network separately
        mu_optimizer = tf.train.AdamOptimizer(learning_rate=mu_learning_rate)
        q_optimizer = tf.train.AdamOptimizer(learning_rate=q_learning_rate)
        self.mu_train_op = mu_optimizer.minimize(
            self.mu_loss, var_list=get_vars('main/mu')
        )
        self.q_train_op = q_optimizer.minimize(
            self.q_loss, var_list=get_vars('main/q')
        )

        # Use soft updates to update the target networks
        self.target_update = tf.group([
            tf.assign(v_targ, decay * v_targ + (1 - decay) * v_main)
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

    def get_action(self, observation, noise_scale):
        action = self.session.run(self.mu, feed_dict={
            self.X: observation.reshape(1,-1)
        })[0]
        action += noise_scale * np.random.randn(self.action_number)
        return np.clip(action, -self.action_max, self.action_max)

    def play(self, random_action=False, render=False):
        done = False
        observation = self.env.reset()
        episode_return = 0
        episode_length = 0
        while not done:
            if random_action:
                action = self.env.action_space.sample()
            else:
                action = self.get_action(observation, self.action_noise)

            next_observation, reward, done, _ = self.env.step(action)

            episode_return += reward
            episode_length += 1

            self.replay_buffer.store(
                observation, action, reward, next_observation,
                False if episode_length == 200 else done
            )
            observation = next_observation
            if render:
                self.env.render()

        return episode_return, episode_length

    def train(
        self,
        episode_number=200,
        exploration_episode_number=50,
        test_agent_every=25,
        save_checkpoint_every=50,
    ):
        total_steps = 0
        q_losses = []
        mu_losses = []
        episode_returns = []

        for e in range(episode_number):
            episode_return, episode_length = self.play(
                random_action=(e < exploration_episode_number)
            )
            if e == exploration_episode_number:
                print ("Start play using agent")
            print("Played Episode %d, Return %d, Length %d" % (
                e, episode_return, episode_length
            ))
            episode_returns.append(episode_return)
            total_steps = total_steps + episode_length

            ql, mul = self.update(episode_length)
            q_losses += ql
            mu_losses += mul

            if e > 0 and e % test_agent_every == 0:
                self.test()

            if e > 0 and e % save_checkpoint_every == 0:
                self.save_checkpoint()
        print("Trained for total %d steps" % total_steps)


    def update(self, episode_length, batch_size=32):
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
        print("Trained %d steps, Latest Q Loss %f, Mu Loss %f" % (
            episode_length, ql, mul
        ))
        return q_losses, mu_losses

    def test(self, episode_number=5):
        for j in range(episode_number):
            episode_return, episode_length = self.play(
                random_action=False, render=True
            )
            print(
                'Test return:', episode_return,
                'Episode Length:', episode_length
            )


    def save_checkpoint(self):
        model_file = os.path.join(self.checkpoint_dir, "ddpg")

        os.makedirs(os.path.basename(model_file), exist_ok=True)
        save_path = self.saver.save(
            self.session,
            save_path=model_file)
        self.replay_buffer.save(
            os.path.join(self.checkpoint_dir, "replay_buffer")
        )
        print ("Saved checkpoint to", save_path)


    def restore_checkpoint(self):
        model_file = os.path.join(self.checkpoint_dir, "ddpg")
        try:
            self.saver.restore(self.session, model_file)
            print ("Restore checkpoint to", model_file)
        except:
            print ("Failed to load checkpoint")


    def cleanup(self):
        tf.reset_default_graph()
        self.env.close()


if __name__ == "__main__":
    env = gym.make("Pendulum-v0")
    env.reset()

    model_file = os.path.join("model", "model.npz")

    ddpg = DDPG(env)
    ddpg.restore_checkpoint()
    ddpg.save_checkpoint()
    ddpg.train(episode_number=200)
    ddpg.save_checkpoint()
    ddpg.cleanup()
