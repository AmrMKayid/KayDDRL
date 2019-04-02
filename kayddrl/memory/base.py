from abc import ABC, abstractmethod
from collections import deque
from typing import Tuple

import numpy as np
import torch

from kayddrl.configs.default import default
from kayddrl.utils import utils


class Memory(ABC):

    def __init__(self, configs=default()):
        self._configs = configs
        self._hparams = configs.memory
        self.device = configs.glob.device
        utils.set_attr(self, self._hparams)

        self.memory = deque(maxlen=self.buffer_size)

    @abstractmethod
    def update(
            self,
            state: np.ndarray,
            action: np.ndarray,
            reward: np.float64,
            next_state: np.ndarray,
            done: float,
    ):
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
    def sample(self) -> Tuple[torch.Tensor, ...]:
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

    def __len__(self) -> int:
        r"""
        :return: the current size of internal memory.
        """
        return len(self.memory)
