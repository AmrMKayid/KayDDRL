from abc import ABC, abstractmethod

from gym import spaces

from kayddrl.utils import utils


class BaseEnv(ABC):

    def __init__(self, config):
        self.done = False
        self._config = config
        utils.set_attr(self, self._config, [
            'name',
            'type',
            'seed',
            'to_render',
            'frame_sleep',
            'max_steps',
            'one_hot',
            'action_bins',
            'reward_scale',
            'num_envs',
        ])

    @abstractmethod
    def reset(self):
        r"""
        Reset the environment
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        r"""
        Take a step in the environment
        :param action:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    def _set_attr_from_u_env(self, u_env):
        r"""
        Set the observation, action dimensions and action type from u_env
        :param u_env:
        :return:
        """
        self.observation_space, self.action_space = self._get_spaces(u_env)
        self.observable_dim = self._get_observable_dim(self.observation_space)
        self.action_dim = self._get_action_dim(self.action_space)
        self.is_discrete = self._is_discrete(self.action_space)

    def _get_spaces(self, u_env):
        r"""
        Helper to set the extra attributes to, and get, observation and action spaces
        :param u_env:
        :return:
        """
        observation_space = u_env.observation_space
        action_space = u_env.action_space
        utils.set_gym_space_attr(observation_space)
        utils.set_gym_space_attr(action_space)
        return observation_space, action_space

    def _get_observable_dim(self, observation_space):
        r"""
        Get the observable dim for an agent in env
        """

        state_dim = observation_space.shape
        if len(state_dim) == 1:
            state_dim = state_dim[0]
        return {'state': state_dim}

    def _get_action_dim(self, action_space):
        r"""
        Get the action dim for an action_space for agent to use
        :param action_space:
        :return:
        """
        if isinstance(action_space, spaces.Box):
            assert len(action_space.shape) == 1
            action_dim = action_space.shape[0]
        elif isinstance(action_space, (spaces.Discrete, spaces.MultiBinary)):
            action_dim = action_space.n
        elif isinstance(action_space, spaces.MultiDiscrete):
            action_dim = action_space.nvec.tolist()
        else:
            raise ValueError('action_space not recognized')
        return action_dim

    def _is_discrete(self, action_space):
        r"""
        Check if an action space is discrete
        :param action_space:
        :return:
        """
        return utils.get_cls_name(action_space) != 'Box'
