import os
from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
import torch

from kayddrl.configs.default import default
from kayddrl.utils.logging import logger


class BaseAgent(ABC):
    r"""
    Abstract Agent used for all agents.
    """

    def __init__(self, env, configs=default()):
        r"""

        :param config:
        :param env:
        """

        self._env = env
        self.env_name = configs.env.name

        self._configs = configs
        self._hparams = configs.agent

        @abstractmethod
        def select_action(self, state: np.ndarray) -> Union[torch.Tensor, np.ndarray]:
            pass

        @abstractmethod
        def step(
                self, action: Union[torch.Tensor, np.ndarray]
        ) -> Tuple[np.ndarray, np.float64, bool]:
            pass

        @abstractmethod
        def update_model(self, *args) -> Tuple[torch.Tensor, ...]:
            pass

        @abstractmethod
        def load_params(self, *args):
            pass

        @abstractmethod
        def save_params(self, params: dict, n_episode: int):
            if not os.path.exists("./saved"):
                os.mkdir("./saved")

            save_name = self.env_name + "_" + self.hparams.name

            path = os.path.join("./saved/" + save_name + "_ep_" + str(n_episode) + ".pt")
            torch.save(params, path)

            logger.info("Saved the model and optimizer to", path)

        @abstractmethod
        def write_log(self, *args):
            pass

        @abstractmethod
        def train(self):
            pass

        def interim_test(self):
            self.args.test = True

            print()
            print("===========")
            print("Start Test!")
            print("===========")

            self._test(interim_test=True)

            print("===========")
            print("Test done!")
            print("===========")
            print()

            self._configs.glob.test = False

        def test(self):
            """Test the agent."""
            self._test()

            # termination
            self.env.close()

        def _test(self, interim_test: bool = False):
            """Common test routine."""

            if interim_test:
                test_num = self._configs.glob.interim_test_nums
            else:
                test_num = self._configs.glob.num_episodes

            for i_episode in range(test_num):
                state = self.env.reset()
                done = False
                score = 0
                step = 0

                while not done:
                    self.env.render()

                    action = self.select_action(state)
                    next_state, reward, done = self.step(action)

                    state = next_state
                    score += reward
                    step += 1

                logger.info(
                    "test %d\tstep: %d\ttotal score: %d" % (i_episode, step, score)
                )

                logger.log_scalar("Test score", score)
