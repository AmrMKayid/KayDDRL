import torch
import pydash as ps
import torch.nn as nn

from kayddrl.comp import models
from kayddrl.networks.base import BaseNN
from kayddrl.utils import utils


class MLP(BaseNN, nn.Module):
    r"""
    MLP with custom hidden layers.

    config example:
    net_config = {
        "name": "NeuralNet",
        "type": "MLP",
        "hid_layers": [
            64
        ],
        "hid_layers_activation": "selu",
        "clip_grad_val": 0.5,
        "loss_config": {
            "name": "MSELoss"
        },
        "optim_config": {
            "name": "Adam",
            "lr": 0.02
        },
        "lr_scheduler_config": {
            "name": "StepLR",
            "step_size": 1000,
            "gamma": 0.9
        },
        "update_type": "polyak",
        "update_frequency": 32,
        "polyak_coef": 0.1,
        "gpu": True
    }
    """

    def __init__(self, config, input_state_size, output_action_size):

        nn.Module.__init__(self)
        super().__init__(config, input_state_size, output_action_size)

        # Default
        utils.set_attr(self, dict(
            out_layer_activation=None,
            init_fn=None,
            clip_grad_val=None,
            loss_config={'name': 'MSELoss'},
            optim_config={'name': 'Adam'},
            lr_scheduler_config=None,
            update_type='replace',
            update_frequency=1,
            polyak_coef=0.0,
            gpu=False,
        ))

        utils.set_attr(self, self._config, [
            'shared',
            'hid_layers',
            'hid_layers_activation',
            'out_layer_activation',
            'init_fn',
            'clip_grad_val',
            'loss_config',
            'optim_config',
            'lr_scheduler_config',
            'update_type',
            'update_frequency',
            'polyak_coef',
            'gpu',
        ])

        dims = [self.input_state_size] + self.hid_layers
        self.model = models.make_fc(dims, self.hid_layers_activation)

        if ps.is_integer(self.output_action_size):
            self.model_tail = models.make_fc([dims[-1], self.output_action_size], self.out_layer_activation)
        else:
            if not ps.is_list(self.out_layer_activation):
                self.out_layer_activation = [self.out_layer_activation] * len(self.output_action_size)
            assert len(self.out_layer_activation) == len(self.output_action_size)
            tails = []
            for out_d, out_activ in zip(self.output_action_size, self.out_layer_activation):
                tail = models.make_fc([dims[-1], out_d], out_activ)
                tails.append(tail)
            self.model_tails = nn.ModuleList(tails)

        models.init_layers(self, self.init_fn)
        self.loss_fn = models.get_loss_fn(self, self.loss_config)
        self.to(self.device)
        self.train()

        # TODO: Logging
        print(utils.describe(self))

    def forward(self, x):
        x = self.model(x)
        if hasattr(self, 'model_tails'):
            outs = []
            for model_tail in self.model_tails:
                outs.append(model_tail(x))
            return outs
        else:
            return self.model_tail(x)
