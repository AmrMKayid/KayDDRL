import time

import gym

from kayddrl.envs.base import BaseEnv
from kayddrl.utils import utils


class GymEnv(BaseEnv):
    r"""
    Basic Gym environment.

    config example:
    "env": {
        "name": "CartPole-v1",
        "type": "openai",
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
        super(GymEnv, self).__init__(config)

        self._env = gym.make(self.name)
        self._set_attr_from_u_env(self._env)
        self._env.seed(self.seed)

        # TODO: Logging
        print(utils.describe(self))

    def reset(self):
        self.done = False
        state = self._env.reset()
        self.render()

        return state

    def step(self, action):
        state, reward, done, info = self._env.step(action)
        self.render()

        return state, reward, done, info

    def render(self):
        if self.to_render:
            self._env.render()
            time.sleep(self.frame_sleep)

    def close(self):
        self._env.close()
