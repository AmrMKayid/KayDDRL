import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from kayddrl.networks.base import BaseNN
from kayddrl.utils import utils
from kayddrl.utils.logging import logger


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class LowDimActor(BaseNN, nn.Module):

    def __init__(self, config, input_state_size, output_action_size, fc1_units, fc2_units):
        nn.Module.__init__(self)
        super(LowDimActor, self).__init__(config, input_state_size, output_action_size)
        self.fc1 = nn.Linear(self.input_state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, self.output_action_size)
        self.reset_parameters()
        self.to(self.device)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class LowDimCritic(BaseNN, nn.Module):
    def __init__(self, config, input_state_size, output_action_size, fc1_units, fc2_units):
        nn.Module.__init__(self)
        super(LowDimCritic, self).__init__(config, input_state_size, output_action_size)
        self.fc1 = nn.Linear(self.input_state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + self.output_action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()
        self.to(self.device)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DDPG_Networks(BaseNN):
    def __init__(self, config, input_state_size, output_action_size, fc1_units=400, fc2_units=300):
        super(DDPG_Networks, self).__init__(config, input_state_size, output_action_size)

        self.actor_local = LowDimActor(config, self.input_state_size,
                                       self.output_action_size, fc1_units, fc2_units)
        self.actor_target = LowDimActor(config, self.input_state_size,
                                        self.output_action_size, fc1_units, fc2_units)

        self.critic_local = LowDimCritic(config, self.input_state_size,
                                         self.output_action_size, fc1_units, fc2_units)
        self.critic_target = LowDimCritic(config, self.input_state_size,
                                          self.output_action_size, fc1_units, fc2_units)

        logger.warn('DDPG_Networks is initialized!')
        logger.info(utils.describe(self))


ddpg = DDPG_Networks({'name': 'DDPG_Config', 'network': {'hidden_size': 32}}, 33, 2)
