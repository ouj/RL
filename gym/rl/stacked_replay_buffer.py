import numpy as np

class StackedFrameReplayBuffer:
    def __init__(
        self,
        frame_width,
        frame_height,
        channels,
        stack_size,
        action_dim,
        max_size = 10000
    ):
        self.current = 0
        self.size = 0
        self.stack_size = stack_size
        self.max_size = max_size
        self.observations = np.zeros(
            [max_size, frame_width, frame_height, channels], dtype=np.float32
        )
        self.actions = np.zeros(
            [max_size, action_dim], dtype=np.float32
        )
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.channels = channels

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
        state = self.observations[index - self.stack_size + 1:index + 1, ...]
        return np.concatenate(state, axis=2)

    def get_valid_indices(self, batch_size):
        indices = np.zeros(batch_size, dtype=int)
        for i in range(batch_size):
            while True:
                # use a lower bound and upper bound to avoid the edge of the
                # ring buffer. We will loss some sample, but is easier to
                # than to dealing with edge cases
                index = np.random.randint(self.stack_size, self.size - 1)
                assert index >= self.stack_size
                assert index != self.size


                if index >= self.current and index < self.current + self.stack_size:
                    # crossing the current pointer
                    continue
                elif self.dones[index - self.stack_size:index].any():
                    # check is there is any done flag in previous state
                    continue
                else:
                    break
            indices[i] = index
        return indices

    def sample_batch(self, batch_size=32):
        assert batch_size <= self.size
        assert self.size >= self.stack_size

        indices = self.get_valid_indices(batch_size=batch_size)
        states_shape = (
            batch_size,
            self.frame_width,
            self.frame_height,
            self.channels * self.stack_size
        )
        previous_states = np.zeros(shape=states_shape, dtype=np.float32)
        states = np.zeros(shape=states_shape, dtype=np.float32)


        for i, index in enumerate(indices):
            previous_states[i, ...] = self.get_state(index - 1)
            states[i, ...] = self.get_state(index)

        return dict(
            s=previous_states,
            a=self.actions[indices],
            r=self.rewards[indices],
            s2=states,
            d=self.dones[indices]
        )
