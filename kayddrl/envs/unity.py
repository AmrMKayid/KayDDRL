import os

import numpy as np
from gym import spaces
from unityagents import brain, UnityEnvironment

from kayddrl.envs.base import BaseEnv
from kayddrl.utils import utils


def get_env_path(env_name):
    r"""
    Get the path to Unity env binaries distributed
    :param env_name:
    :return:
    """
    env_path = utils.smart_path(f'envs/{env_name}-env/build/{env_name}')
    env_dir = os.path.dirname(env_path)
    assert os.path.exists(env_dir), f'Missing {env_path}. See README to run unity env.'
    return env_path


class BrainExt:
    r"""
    Unity Brain class extension, where self = brain    
    """

    def is_discrete(self):
        return self.vector_action_space_type == 'discrete'

    def get_action_dim(self):
        return self.vector_action_space_size

    def get_observable_types(self):
        '''What channels are observable: state, image, sound, touch, etc.'''
        observable = {
            'state': self.vector_observation_space_size > 0,
            'image': self.number_visual_observations > 0,
        }
        return observable

    def get_observable_dim(self):
        '''Get observable dimensions'''
        observable_dim = {
            'state': self.vector_observation_space_size,
            'image': 'some np array shape, as opposed to what Arthur called size',
        }
        return observable_dim


utils.monkey_patch(brain.BrainParameters, BrainExt)


class UnityEnv(BaseEnv):
    r"""
    Basic Unity ML Agent environment.

    config example:
    "env": {
        "name": "Reacher",
        "type": "unity",
        "seed": 0,
        "to_render": True,
        "frame_sleep": 0.001,
        "max_steps": 1000,
        "one_hot": None,
        "action_bins": None,
        "reward_scale": None,
        "num_envs": None,
    }
    """

    def __init__(self, config):
        super(UnityEnv, self).__init__(config)

        self._env = UnityEnvironment(file_name=get_env_path(self.name), seed=self.seed)
        self.patch_gym_spaces(self._env)
        self._set_attr_from_u_env(self._env)

        # TODO: Logging
        print(utils.describe(self))

    def reset(self):
        self.done = False
        info_dict = self._env.reset(train_mode=self.to_render)
        env_info = self._get_env_info(info_dict, 0)
        state = env_info.vector_observations[0]
        return state

    def step(self, action):
        info_dict = self._env.step(action)
        env_info = self._get_env_info(info_dict, 0)
        state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return state, reward, done, env_info

    def close(self):
        self._env.close()

    def _get_brain(self, env, brain_index):
        r"""
        Get the unity-equivalent of agent, i.e. brain, to access its info
        :param env:
        :param brain_index:
        :return:
        """
        brain_name = env.brain_names[brain_index]
        brain = env.brains[brain_name]
        return brain

    def patch_gym_spaces(self, env):
        r"""
        For standardization, use gym spaces to represent observation and action spaces for Unity.
        This method iterates through the multiple brains (multiagent) then constructs and returns lists of observation_spaces and action_spaces
        :param env:
        :return:
        """

        observation_spaces = []
        action_spaces = []
        for brain_index in range(len(env.brain_names)):
            brain = self._get_brain(env, brain_index)

            # TODO: Logging
            utils.describe(brain)

            observation_shape = (brain.get_observable_dim()['state'],)
            action_dim = (brain.get_action_dim(),)

            if brain.is_discrete():
                dtype = np.int32
                action_space = spaces.Discrete(brain.get_action_dim())
            else:
                dtype = np.float32
                action_space = spaces.Box(low=0.0, high=1.0, shape=action_dim, dtype=dtype)

            observation_space = spaces.Box(low=0, high=1, shape=observation_shape, dtype=dtype)
            utils.set_gym_space_attr(observation_space)
            utils.set_gym_space_attr(action_space)
            observation_spaces.append(observation_space)
            action_spaces.append(action_space)

        # set for singleton
        env.observation_space = observation_spaces[0]
        env.action_space = action_spaces[0]

        return observation_spaces, action_spaces

    def _get_env_info(self, env_info_dict, index):
        r"""
        Unity API returns a env_info_dict. Use this method to pull brain(env)-specific
        :param env_info_dict:
        :param index:
        :return:
        """
        brain_name = self._env.brain_names[index]
        env_info = env_info_dict[brain_name]
        return env_info
