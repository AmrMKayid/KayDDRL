from torch import distributions

from kayddrl.comp import distribution

setattr(distributions, 'Argmax', distribution.Argmax)
setattr(distributions, 'GumbelCategorical', distribution.GumbelCategorical)
setattr(distributions, 'MultiCategorical', distribution.MultiCategorical)

# Probability Distributions constraints for different action types;
# the first in the list is the default
ACTION_PDS = {
    'continuous': ['Normal', 'Beta', 'Gumbel', 'LogNormal'],
    'multi_continuous': ['MultivariateNormal'],
    'discrete': ['Categorical', 'Argmax', 'GumbelCategorical'],
    'multi_discrete': ['MultiCategorical'],
    'multi_binary': ['Bernoulli'],
}


# -------------------- Base Methods for Action Policy -------------------- #

def get_action_pd_cls(action_prob_dist_type, action_type):
    r"""
    Verify and get the action prob. distribution class for construction
    :param action_prob_dist_type: type of action prob. distribution
    :param action_type: type of action space
    :return:
    """
    pdtypes = ACTION_PDS[action_type]
    assert action_prob_dist_type in pdtypes, \
        f'Pdtype {action_prob_dist_type} is not compatible/supported with action_type {action_type}. Options are: {pdtypes}'
    ActionProbDist = getattr(distributions, action_prob_dist_type)
    return ActionProbDist
