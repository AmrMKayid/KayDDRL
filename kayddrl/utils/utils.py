import torch
import random
import numpy as np
import pydash as ps

from pprint import pformat


# -------------------- Seeding -------------------- #

def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -------------------- ---------- -------------------- #


# -------------------- Logging -------------------- #


def get_cls_name(obj, lower=False):
    r"""
    Get the class name of an object
    """

    class_name = obj.__class__.__name__
    if lower:
        class_name = class_name.lower()
    return class_name


def get_cls_attr(obj):
    r"""
    Get the class attr of an object as dict
    """
    attr_dict = {}
    for k, v in obj.__dict__.items():
        if hasattr(v, '__dict__'):
            val = str(v)
        else:
            val = v
        attr_dict[k] = val
    return attr_dict


def describe(cls):
    desc_list = [f'{get_cls_name(cls)}:']
    for k, v in get_cls_attr(cls).items():
        if k == 'config':
            desc_v = v['name']
        elif ps.is_dict(v) or ps.is_dict(ps.head(v)):
            desc_v = pformat(v)
        else:
            desc_v = v
        desc_list.append(f'- {k} = {desc_v}')
    desc = '\n'.join(desc_list)
    return desc


# -------------------- ---------- -------------------- #


# -------------------- Data Types -------------------- #


def numpify(x, dtype):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(dtype)
    elif isinstance(x, np.ndarray):
        return x.astype(dtype)
    else:
        return np.asarray(x, dtype=dtype)


def tensorify(x, device):
    if torch.is_tensor(x):
        if str(x.device) != str(device):
            x = x.to(device)
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    else:
        return torch.from_numpy(np.asarray(x)).float().to(device)


# -------------------- ---------- -------------------- #

def set_attr(obj, attr_dict, keys=None):
    r"""
    Set attribute of an object from a dict
    :param obj:
    :param attr_dict:
    :param keys:
    :return:
    """

    if keys is not None:
        attr_dict = ps.pick(attr_dict, keys)
    for attr, val in attr_dict.items():
        setattr(obj, attr, val)
    return obj
