import numpy as np

class StackedReplayBuffer:
    def __init__(
        self,
        observation_shape,
        action_dim,
        stack_size,
        max_size = 1000000
    ):
        self.current = 0
        self.size = 0
        self.stack_size = stack_size
        self.max_size = max_size
        self.observations = np.zeros(
            [max_size, *observation_shape], dtype=np.float32
        )
        self.actions = np.zeros(
            [max_size, action_dim], dtype=np.float32
        )
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)

    def number_of_samples(self):
        return self.size

    def store(self, observation, action, reward, done):
        self.observations[self.current] = observation
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.dones[self.current] = done
        self.current = (self.current + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def get_state(self, index):
        if self.size == 0:
            raise ValueError("Empty replay buffer")
        if index < self.stack_size - 1:
            raise ValueError(
                "For stack of %d, the minimal index must be %d" % (
                    self.stack_size, self.stack_size - 1
                )
            )
        return self.observations[index - self.stack_size + 1:index+1, ...]

    def get_valid_indices(self, batch_size):
        indices = np.zeros(batch_size)
        for i in range(batch_size):
            while True:
                index = np.random.randint(0, self.size - 1)
                if index < self.stack_size:
                    continue
                elif index >= self.current and index - self.stack_size <= self.current:
                    continue
                elif self.dones[index - self.stack_size:index].any():
                    continue
                else:
                    break
            indices[i] = index
        return indices

    def sample_batch(self, batch_size=32):
        assert batch_size <= self.size
        assert self.size >= self.stack_size

        indexes = self.get_valid_indices(batch_size=batch_size)

        return dict(
            s=self.observations[indexes],
            a=self.actions[indexes],
            r=self.rewards[indexes],
            d=self.dones[indexes]
        )
