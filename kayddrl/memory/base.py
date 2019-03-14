from abc import ABC, abstractmethod
from collections import namedtuple, deque

from kayddrl.utils import utils


class Memory(ABC):

    def __init__(self, config):
        self._config = config
        utils.set_attr(self, self._config, {
            'batch_size',
            'buffer_size',
        })
        self.memory = deque(maxlen=self.buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

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
        Memory sampling mechanism
        :return: batch of experiences
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self):
        r"""
        Method to fully reset the memory storage and related variables
        """
        raise NotImplementedError

    def __len__(self):
        r"""
        :return: the current size of internal memory.
        """
        return len(self.memory)
