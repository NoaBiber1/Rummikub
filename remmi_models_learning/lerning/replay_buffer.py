import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, x, reward, next_pos_x, done):
        x = x.detach() if hasattr(x, "detach") else x
        next_pos_x = [s.detach() if hasattr(s, "detach") else s for s in next_pos_x]
        self.buffer.append((x, reward, next_pos_x, done))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)