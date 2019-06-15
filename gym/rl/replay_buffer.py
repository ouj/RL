import numpy as np

class ReplayBuffer:
    def __init__(
        self,
        observation_shape,
        action_shape,
        max_size = 1000000
    ):
        self.index = 0
        self.size = 0
        self.max_size = max_size
        self.observations = np.zeros(
            [max_size, *observation_shape], dtype=np.float32
        )
        self.next_observations = np.zeros(
            [max_size, *observation_shape], dtype=np.float32
        )
        self.actions = np.zeros(
            [max_size, *action_shape], dtype=np.float32
        )
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)

    def number_of_samples(self):
        return self.size

    def store(self, observation, action, reward, next_observation, done):
        self.observations[self.index] = observation
        self.next_observations[self.index] = next_observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        assert batch_size <= self.size
        indexes = np.random.randint(0, self.size, size=batch_size)
        return dict(
            s=self.observations[indexes],
            s2=self.next_observations[indexes],
            a=self.actions[indexes],
            r=self.rewards[indexes],
            d=self.dones[indexes]
        )

    def save(self, filename):
        params = {
            "index": self.index,
            "size": self.size,
            "max_size": self.max_size,
            "observations": self.observations,
            "next_observations": self.next_observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
        }
        np.savez(filename, *params)

    def load(self, filename):
        params = np.load(filename)
        self.index = params["index"]
        self.size = params["size"]
        self.max_size = params["max_size"]
        self.observations = params["observation"]
        self.actions = params["action"]
        self.rewards = params["rewards"]
        self.dones = params["dones"]
