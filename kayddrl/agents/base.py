import torch
import numpy as np
import pydash as ps

from kayddrl import memory
from kayddrl.agents import algos
from kayddrl.utils import utils


class BaseAgent:

    def __init__(self, config):
        self._config = config

        memory_cls = getattr(memory, ps.get(self._config, 'memory.name'))
        self.memory = memory_cls(self._config['memory'])
        algorithm_cls = getattr(algos, ps.get(self._config, 'algo.name'))
        self.algo = algorithm_cls(self)

        # TODO: Logging
        print(utils.describe(self))

    def observe(self, state, action, reward, next_state, done):
        self.memory.update(state, action, reward, next_state, done)
        loss = self.algo.train()
        if not np.isnan(loss):
            self.loss = loss
        explore_var = self.algo.update()
        return loss, explore_var

    def act(self, state):
        with torch.no_grad():
            action = self.algo.act(state)
        return action

    def learn(self, experiences):
        raise NotImplementedError

    def save(self, ckpt=None):
        self.algo.save(ckpt=ckpt)

    def close(self):
        self.save()
