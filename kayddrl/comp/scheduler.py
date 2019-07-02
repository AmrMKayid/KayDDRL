import numpy as np

from kayddrl.comp import compute
from kayddrl.utils import utils


class VarScheduler:
    '''
    Variable scheduler for decaying variables such as explore_var (epsilon, tau) and entropy

    e.g. config
    "var_config": {
        "name": "linear_decay",
        "start_val": 1.0,
        "end_val": 0.1,
        "start_step": 0,
        "end_step": 800,
    },
    '''

    def __init__(self, var_decay_config=None):
        self._updater_name = 'no_decay' if var_decay_config is None else var_decay_config['name']
        self._updater = getattr(compute, self._updater_name)
        utils.set_attr(self, dict(
            start_val=np.nan,
        ))
        utils.set_attr(self, var_decay_config, [
            'start_val',
            'end_val',
            'start_step',
            'end_step',
        ])
        if not getattr(self, 'end_val', None):
            self.end_val = self.start_val

    def update(self, step):
        r"""
        Get an updated value for var
        :param step:
        :return:
        """

        if self._updater_name == 'no_decay':
            return self.end_val
        val = self._updater(self.start_val, self.end_val, self.start_step, self.end_step, step)
        return val
