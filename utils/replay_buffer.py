import random
from collections import deque

class ReplayBuffer:
    def __init__(self, size: int):
        self.memory = deque(maxlen=size)

    def __len__(self):
        return len(self.memory)

    def push(self, ex):
        self.memory.append(ex)

    def sample(self, batch_size):
        return random.sample(self.memory, k=batch_size)
