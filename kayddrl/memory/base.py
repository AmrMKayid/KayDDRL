from abc import ABC, abstractmethod
from collections import namedtuple, deque

from kayddrl.utils import utils


class Memory(ABC):

    def __init__(self, config):
        self._config = config
        utils.set_attr(self, self._config, [
            'batch_size',
            'buffer_size',
            'device',
        ])
        self.memory = deque(maxlen=self.buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        # TODO: Logging
        print(utils.describe(self))

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        r"""
        Update memory with new experience
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        r"""
        Memory sampling mechanism, Randomly sample a batch of experiences from memory.
        :return: batch of experiences
        """
        raise NotImplementedError

    def clear(self):
        r"""
        Method to fully reset the memory storage and related variables
        """
        self.memory.clear()

    def __len__(self):
        r"""
        :return: the current size of internal memory.
        """
        return len(self.memory)
