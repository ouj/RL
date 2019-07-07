# Replay Buffer
import numpy as np

class SimpleReplayBuffer:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        buffer_size = len(self.buffer)
        indices = np.random.choice(
            np.arange(buffer_size), size=batch_size, replace=False
        )
        size = len(indices)
        states = np.zeros([size, 210, 160, 4], dtype=np.float32)
        next_states = np.zeros([size, 210, 160, 4], dtype=np.float32)
        actions = np.zeros(size, dtype=np.float32)
        rewards = np.zeros(size, dtype=np.float32)
        dones = np.zeros(size, dtype=np.float32)

        for i, idx in enumerate(indices):
            state, action, reward, next_state, done = self.buffer[idx]
            states[i] = state
            next_states[i] = next_state
            actions[i] = action
            rewards[i] = reward
            dones[i] = done

        return dict(s=states, s2=next_states, a=actions, r=rewards, d=dones)

    def store(self, state, action, reward, next_state, done):
        self.add((state, action, reward, next_state, done))

    def number_of_samples(self):
        return len(self.buffer)
