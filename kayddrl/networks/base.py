import torch
import pydash as ps
import torch.nn as nn

from abc import ABC

from kayddrl.comp import old_models


class BaseNN(ABC):

    def __init__(self, config, input_state_size, output_action_size):

        self._config = config
        self.input_state_size = input_state_size
        self.output_action_size = output_action_size

        if self._config.get('gpu'):
            if torch.cuda.device_count():
                self.device = f'cuda:{config.get("cuda_id", 0)}'
            else:
                self.device = 'cpu'
        else:
            self.device = 'cpu'
