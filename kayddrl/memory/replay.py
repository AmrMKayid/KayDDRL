import random
import numpy as np

from kayddrl.memory.base import Memory


class ReplayBuffer(Memory):
    r"""
    Stores agent experiences and samples from them for agent training

    An experience consists of
        - state: representation of a state
        - action: action taken
        - reward: scalar value
        - next state: representation of next state (should be same as state)
        - done: 0 / 1 representing if the current state is the last in an episode

    The memory has a size of N. When capacity is reached, the oldest experience
    is deleted to make space for the latest experience.
        - This is implemented as a circular buffer so that inserting experiences are O(1)
        - Each element of an experience is stored as a separate array of size N * element dim

    When a batch of experiences is requested, K experiences are sampled according to a random uniform distribution.

    config example:
    "memory": {
        "name": "ReplayBuffer",
        "batch_size": 32,
        "buffer_size": 10000,
    }
    """

    def __init__(self, config):
        super().__init__(config)
        self.clear()

    def update(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        sample_size = self.batch_size if self.batch_size <= len(self) else len(self)
        experiences = random.sample(self.memory, k=sample_size)

        states = np.array([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None])
        rewards = np.array([e.reward for e in experiences if e is not None])
        next_states = np.array([e.next_state for e in experiences if e is not None])
        dones = np.array([e.done for e in experiences if e is not None])

        return (states, actions, rewards, next_states, dones)
