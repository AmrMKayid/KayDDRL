import torch
import pydash as ps
import torch.nn as nn

from abc import ABC

from kayddrl.comp import old_models


class BaseNN(ABC):

    def __init__(self, config, input_dim, output_dim):

        self._config = config
        self.input_dim = input_dim
        self.output_dim = output_dim

        if self._config.get('gpu'):
            if torch.cuda.device_count():
                self.device = f'cuda:{config.get("cuda_id", 0)}'
            else:
                self.device = 'cpu'
        else:
            self.device = 'cpu'
