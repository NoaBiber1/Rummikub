"""Fixed-capacity FIFO replay buffer."""

import random
from collections import deque


class ReplayBuffer:
    """Bounded FIFO of (x, reward, next_pos_x, done), sampled uniformly."""

    def __init__(self, capacity):
        """A buffer holding at most `capacity` transitions."""
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, x, reward, next_pos_x, done):
        """Store one transition, detaching any tensors on the way in."""
        x = x.detach() if hasattr(x, "detach") else x
        next_pos_x = [s.detach() if hasattr(s, "detach") else s for s in next_pos_x]
        self.buffer.append((x, reward, next_pos_x, done))

    def sample(self, batch_size):
        """A uniform random batch, clamped to what the buffer holds."""
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Number of stored transitions."""
        return len(self.buffer)