import os
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
