from abc import ABC, abstractmethod

from kayddrl.utils import utils


class BaseEnv(ABC):

    def __init__(self, config):
        self._config = config
        utils.set_attr(self, self._config, [
            'name',
            'seed',
            'max_steps',
            'one_hot',
            'action_bins',
            'reward_scale',
            'num_envs',
        ])

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abstractmethod
    def render(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError
