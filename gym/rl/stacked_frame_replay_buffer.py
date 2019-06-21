import numpy as np


class StackedFrameReplayBuffer:
    def __init__(
        self,
        frame_height,
        frame_width,
        stack_size,
        action_dim,
        batch_size=32,
        max_size=10000,
    ):
        self.current = 0
        self.size = 0
        self.stack_size = stack_size
        self.max_size = max_size
        self.frames = np.zeros([max_size, frame_height, frame_width], dtype=np.float32)
        self.actions = np.zeros([max_size, action_dim], dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.batch_size = batch_size
        self.indices = np.zeros(batch_size, dtype=int)

        states_shape = (
            self.batch_size,
            self.frame_height,
            self.frame_width,
            self.stack_size,
        )
        self.previous_states = np.zeros(shape=states_shape, dtype=np.float32)
        self.states = np.zeros(shape=states_shape, dtype=np.float32)

    def number_of_samples(self):
        return self.size

    def store(self, frame, action, reward, done):
        self.frames[self.current] = frame
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
                "For stack of %d, the minimal index must be %d"
                % (self.stack_size, self.stack_size - 1)
            )
        state = self.frames[index - self.stack_size + 1 : index + 1, ...]
        return np.stack(state, axis=2)

    def get_valid_indices(self):
        for i in range(self.batch_size):
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
                elif self.dones[index - self.stack_size : index].any():
                    # check is there is any done flag in previous state
                    continue
                else:
                    break
            self.indices[i] = index

    def sample_batch(self):
        assert self.batch_size <= self.size
        assert self.size >= self.stack_size

        self.get_valid_indices()
        for i, index in enumerate(self.indices):
            self.previous_states[i, ...] = self.get_state(index - 1)
            self.states[i, ...] = self.get_state(index)

        return dict(
            s=self.previous_states,
            a=self.actions[self.indices],
            r=self.rewards[self.indices],
            s2=self.states,
            d=self.dones[self.indices],
        )
